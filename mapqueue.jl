using SparseMatrix
using SparseArrays
using NMarkov
using Plots
using Origin
using SymbolicDiff
using SymbolicMarkov

using LinearAlgebra
using Origin: @origin
using LinearAlgebra.BLAS: gemv!, scal!, axpy!, copy!
using SparseMatrix: spdiag, spger!
using NMarkov: itime, @dot, rightbound, poipmf!, unif

using Distributions
using Random

eye(T::Type, m, n) = Matrix{T}(I, m, n)

"""
compute: v1 = v0 * exp(Q*t)
"""

function fexp!(P, qv, t, v0, v1; eps, poi, xtmp, tmpv)
    right = rightbound(qv * t, eps) + 1
    weight = poipmf!(qv * t, poi, left=0, right=right)
    @. v1 = 0.0
    @. xtmp = v0
    @origin (poi => 0) begin
        axpy!(poi[0], xtmp, v1)
        for i = 1:right
            gemv!('T', 1.0, P, xtmp, false, tmpv)
            @. xtmp = tmpv
            axpy!(poi[i], xtmp, v1)
        end
    end
    scal!(1/weight, v1)
end

"""
compute: v1 = exp(Q*t) * v0
"""

function bexp!(P, qv, t, v0, v1; eps, poi, xtmp, tmpv)
    right = rightbound(qv * t, eps) + 1
    weight = poipmf!(qv * t, poi, left=0, right=right)
    @. v1 = 0.0
    @. xtmp = v0
    @origin (poi => 0) begin
        axpy!(poi[0], xtmp, v1)
        for i = 1:right
            gemv!('N', 1.0, P, xtmp, false, tmpv)
            @. xtmp = tmpv
            axpy!(poi[i], xtmp, v1)
        end
    end
    scal!(1/weight, v1)
end

"""
compute: en += int_0^t exp(Q*s) * b * f * exp(Q*(t-s)) ds
"""

function conv!(P, qv, t, f, b, en; scaling=1.0, eps, poi, xtmp, tmpv, vx)
    right = rightbound(qv * t, eps) + 1
    weight = poipmf!(qv * t, poi, left=0, right=right)
    @origin (poi=>0) begin
        @. vx[right] = 0.0
        axpy!(poi[right], b, vx[right])
        for l = right-1:-1:1
            @. vx[l] = 0.0
            gemv!('N', 1.0, P, vx[l+1], false, vx[l])
            axpy!(poi[l], b, vx[l])
        end
    end
    @. xtmp = f
    spger!(scaling/(qv*weight), xtmp, vx[1], 1.0, en)
    for l = 1:right-1
        @. tmpv = 0.0
        gemv!('T', 1.0, P, xtmp, false, tmpv)
        @. xtmp = tmpv
        spger!(scaling/(qv*weight), xtmp, vx[l+1], 1.0, en)
    end
end

"""
udat: time series data for ulitization
t0: time length of unobserved period
t1: time length of observed period


L = alpha * exp(Q * t0) * V[1] * exp(Q * t0) * V[2] * exp(Q * t0) * V[3] * ... * exp(Q * t0) * V[K] * xi

V[k] = (Lam0 * exp(Q00 * t1d) * Q01 * exp(Q11 * t1u) * Lam1' + Lam1 * exp(Q11 * t1u) * Q10 * exp(Q00 * t1d) * Lam0')
V[k] = (Lam0 * exp(Q00 * t1d) * Lam0') where t1d = t1
V[k] = (Lam1 * exp(Q11 * t1u) * Lam1') where t1u = t1

====
vf0[k] = alpha * exp(Q * t0) * V[1] * ... * exp(Q * t0) * V[k]

vf0[0] = alpha
vf0[1] = alpha * exp(Q * t0) * V[1]
...
vf0[K] = alpha ... V[K]
vf0[K+1] does not exist

====
vf1[k] = alpha * exp(Q * t0) * V[1] * ... * exp(Q * t0) * V[k] * exp(Q * t0)

vf1[0] = alpha * exp(Q * t0)
vf1[1] = alpha * exp(Q * t0) * V[1] * exp(Q * t0)
...
vf1[K-1] = alpha * ... * V[K-1] * exp(Q * t0)
vf1[K] does not exist

====
vb0[k] = exp(Q * t0) * V[k] * ... * exp(Q * t0) * V[K] * xi

vb0[0] does not exist
vb0[1] = exp(Q * t0) * V[1] * ... * xi
...
vb0[K] = exp(Q * t0) * V[K] * xi
vb0[K+1] = xi

====
vb1[k] = V[k] * exp(Q * t0) * V[k+1] * ... * exp(Q * t0) * V[K] * xi

vb1[0] does not exist
vb1[1] = V[1] * exp(Q * t0) * ... * xi
...
vb1[K] = V[K] * xi
vb1[K+1] does not exist

L = vf0[k] * vb0[k+1]
L = vf1[k] * vb1[k+1]
"""

@origin (vf0=>0, vf1=>0, vb0=>0, vb1=>0, fw=>0, bw=>0) function estep!(
        alpha, Q, Q00, Q01, Q10, Q11, Lam0, Lam1, xi,　nn0, nn1,
        t0, t1, udat, vf0, vf1, vb0, vb1, fw, bw, eb, en, en00, en01, en10, en11; eps = 1.0e-8, ufactor = 1.01)

    K = length(t0)
    P, qv = NMarkov.unif(Q, ufactor)
    P00 = P[nn0,nn0]
    P11 = P[nn1,nn1]
    
    maxt = max(maximum(t0), maximum(t1))
    maxright = rightbound(qv * maxt, eps) + 1
    poi = zeros(maxright+1)

    n, _ = size(Q)
    n0, _ = size(Q00)
    n1, _ = size(Q11)
    xtmp = zeros(n)
    tmpv = zeros(n)

    xtmp0 = zeros(n0)
    tmpv0 = zeros(n0)
    vftmp00 = zeros(n0)
    vftmp01 = zeros(n0)
    vbtmp00 = zeros(n0)
    vbtmp01 = zeros(n0)

    xtmp1 = zeros(n1)
    tmpv1 = zeros(n1)
    vftmp10 = zeros(n1)
    vftmp11 = zeros(n1)
    vbtmp10 = zeros(n1)
    vbtmp11 = zeros(n1)
    
    vx = Vector{Vector{Float64}}(undef, maxright+1)
    vx0 = Vector{Vector{Float64}}(undef, maxright+1)
    vx1 = Vector{Vector{Float64}}(undef, maxright+1)
    for i = 1:maxright+1
        vx[i] = zeros(n)
        vx0[i] = zeros(n0)
        vx1[i] = zeros(n1)
    end
    
    # forward: compute vf0 and vf1
    copy!(vf0[0], alpha)
    for k = 1:K
        # vf1[k-1] = vf0[k-1] * exp(Q * t0[k])
        fexp!(P, qv, t0[k], vf0[k-1], vf1[k-1]; eps=eps, poi=poi, xtmp=xtmp, tmpv=tmpv)
#         println(vf1[k-1])

        # vf0[k] = vf1[k-1] * V[k]
        @. vf0[k] = 0.0
        if udat[k] == 0.0
            # V[k] = Lam0 * exp(Q00 * t1[k]) * Lam0'
            gemv!('T', 1.0, Lam0, vf1[k-1], false, vftmp00)
            fexp!(P00, qv, t1[k], vftmp00, vftmp01; eps=eps, poi=poi, xtmp=xtmp0, tmpv=tmpv0)
            gemv!('N', 1.0, Lam0, vftmp01, false, vf0[k])
        elseif udat[k] == 1.0
            gemv!('T', 1.0, Lam1, vf1[k-1], false, vftmp10)
            fexp!(P11, qv, t1[k], vftmp10, vftmp11; eps=eps, poi=poi, xtmp=xtmp1, tmpv=tmpv1)
            gemv!('N', 1.0, Lam1, vftmp11, false, vf0[k])
        else
            gemv!('T', 1.0, Lam0, vf1[k-1], false, vftmp00)
#             println("vftmp00", vftmp00)
            fexp!(P00, qv, t1[k]*(1-udat[k]), vftmp00, vftmp01; eps=eps, poi=poi, xtmp=xtmp0, tmpv=tmpv0)
#             println("vftmp01", vftmp01)
            gemv!('T', 1.0, Q01, vftmp01, false, vftmp10)
#             println("vftmp10", vftmp10)
            fexp!(P11, qv, t1[k]*udat[k], vftmp10, vftmp11; eps=eps, poi=poi, xtmp=xtmp1, tmpv=tmpv1)
#             println("vftmp11", vftmp11)
            gemv!('N', 1.0, Lam1, vftmp11, false, vf0[k])
#             println("vf0", vf0[k])

            gemv!('T', 1.0, Lam1, vf1[k-1], false, vftmp10)
            fexp!(P11, qv, t1[k]*udat[k], vftmp10, vftmp11; eps=eps, poi=poi, xtmp=xtmp1, tmpv=tmpv1)
            gemv!('T', 1.0, Q10, vftmp11, false, vftmp00)
            fexp!(P00, qv, t1[k]*(1-udat[k]), vftmp00, vftmp01; eps=eps, poi=poi, xtmp=xtmp0, tmpv=tmpv0)
            gemv!('N', 1.0, Lam0, vftmp01, 1.0, vf0[k]) # vf0[k] += ...
        end
        ## scaling
        tmp = sum(vf0[k])
        scal!(1/tmp, vf0[k])
        fw[k] = fw[k-1] + log(tmp)
#         println(fw[k])
    end

    # backward: compute vb0 and vb1
    copy!(vb0[K+1], xi)
    for k = K:-1:1
        # vb1[k] = V[k] * vb0[k+1]
        @. vb1[k] = 0.0
        if udat[k] == 0.0
            # V[k] = Lam0 * exp(Q00 * t1[k]) * Lam0'
            gemv!('T', 1.0, Lam0, vb0[k+1], false, vftmp00)
            bexp!(P00, qv, t1[k], vftmp00, vftmp01; eps=eps, poi=poi, xtmp=xtmp0, tmpv=tmpv0)
            gemv!('N', 1.0, Lam0, vftmp01, false, vb1[k])
        elseif udat[k] == 1.0
            gemv!('T', 1.0, Lam1, vb0[k+1], false, vftmp10)
            bexp!(P11, qv, t1[k], vftmp10, vftmp11; eps=eps, poi=poi, xtmp=xtmp1, tmpv=tmpv1)
            gemv!('N', 1.0, Lam1, vftmp11, false, vb1[k])
        else
            gemv!('T', 1.0, Lam0, vb0[k+1], false, vftmp00)
            bexp!(P00, qv, t1[k]*(1-udat[k]), vftmp00, vftmp01; eps=eps, poi=poi, xtmp=xtmp0, tmpv=tmpv0)
            gemv!('N', 1.0, Q10, vftmp01, false, vftmp10)
            bexp!(P11, qv, t1[k]*udat[k], vftmp10, vftmp11; eps=eps, poi=poi, xtmp=xtmp1, tmpv=tmpv1)
            gemv!('N', 1.0, Lam1, vftmp11, false, vb1[k])

            gemv!('T', 1.0, Lam1, vb0[k+1], false, vftmp10)
            bexp!(P11, qv, t1[k]*udat[k], vftmp10, vftmp11; eps=eps, poi=poi, xtmp=xtmp1, tmpv=tmpv1)
            gemv!('N', 1.0, Q01, vftmp11, false, vftmp00)
            bexp!(P00, qv, t1[k]*(1-udat[k]), vftmp00, vftmp01; eps=eps, poi=poi, xtmp=xtmp0, tmpv=tmpv0)
            gemv!('N', 1.0, Lam0, vftmp01, 1.0, vb1[k]) # vb1[k] += ...
        end
        
        # vb0[k] = exp(Q * t0[k]) * vb1[k]
        bexp!(P, qv, t0[k], vb1[k], vb0[k]; eps=eps, poi=poi, xtmp=xtmp, tmpv=tmpv)
        
        # scaling
        tmp = sum(vb0[k])
        scal!(1/tmp, vb0[k])
        bw[k] = bw[k+1] + log(tmp)
#         println(bw[k])
    end
    
    # computation
    @. eb = alpha * vb0[1]
    tmp = sum(eb)
    llf = log(tmp) + bw[1]
    scal!(1/tmp, eb)
    
    # en
    #   conv!(Q, t0[k], vf0[k-1], vb1[k], en)
    #   if u[k] = 0
    #      tmpf00 = vf1[k-1] * Lam0
    #      tmpb00 = Lam0 * vb0[k+1]
    #      conv!(Q00, t1[k], tmpf00, tmpb00, en00)
    #   elseif u[k] = 1
    #      tmpf11 = vf1[k-1] * Lam1
    #      tmpb11 = Lam1 * vb0[k+1]
    #      conv!(Q11, t1[k], tmpf11, tmpb11, en11)
    #   else
    #      tmpf00 = vf1[k-1] * Lam0
    #      tmpf01 = tmpf00 * exp(Q00 * t1 * (1-u[k]))
    #      tmpf10 = tmpf01 * Q01
    #      tmpb11 = Lam1 * vb1[k+1]
    #      tmpb10 = exp(Q11 * t1 * u[k]) * tmpb11
    #      tmpb01 = Q01 * tmpb10
    #      conv!(Q00, t1 * (1-u[k]), tmpf00, tmpb01, en00)
    #      conv!(Q11, t1 * u[k], tmpf10, tmpb11, en11)
    #      spger!(tmpf01, tmpb10, en01)
    #
    #      tmpf10 = vf1[k-1] * Lam1
    #      tmpf11 = tmpf10 * exp(Q11 * t1 * u[k])
    #      tmpf00 = tmpf11 * Q10
    #      tmpb01 = Lam0 * vb1[k+1]
    #      tmpb00 = exp(Q00 * t1 * (1-u[k])) * tmpb01
    #      tmpb11 = Q10 * tmpb00
    #      conv!(Q11, t1 * u[k], tmpf10, tmpb11, en11)
    #      conv!(Q00, t1 * (1-u[k]), tmpf00, tmpb01, en00)
    #      spger!(tmpf11, tmpb00, en10)
    #   end

#     @. en.val = 0.0
#     @. en00.val = 0.0
#     @. en01.val = 0.0
#     @. en10.val = 0.0
#     @. en11.val = 0.0
    @. en.nzval = 0.0
    @. en00.nzval = 0.0
    @. en01.nzval = 0.0
    @. en10.nzval = 0.0
    @. en11.nzval = 0.0
    
    for k = 1:K
        conv!(P, qv, t0[k], vf0[k-1], vb1[k], en;
            scaling=exp(fw[k-1]+bw[k+1]-llf), eps=eps, poi=poi, xtmp=xtmp, tmpv=tmpv, vx=vx)
        if udat[k] == 0.0
            gemv!('T', 1.0, Lam0, vf1[k-1], false, vftmp00)
            gemv!('T', 1.0, Lam0, vb0[k+1], false, vbtmp00)
            conv!(P00, qv, t1[k], vftmp00, vbtmp00, en00;
                scaling=exp(fw[k-1]+bw[k+1]-llf), eps=eps, poi=poi, xtmp=xtmp0, tmpv=tmpv0, vx=vx0)
        elseif udat[k] == 1.0
            gemv!('T', 1.0, Lam1, vf1[k-1], false, vftmp11)
            gemv!('T', 1.0, Lam1, vb0[k+1], false, vbtmp11)
            conv!(P11, qv, t1[k], vftmp11, vbtmp11, en11;
                scaling=exp(fw[k-1]+bw[k+1]-llf), eps=eps, poi=poi, xtmp=xtmp1, tmpv=tmpv1, vx=vx1)
        else
            gemv!('T', 1.0, Lam0, vf1[k-1], false, vftmp00)
            fexp!(P00, qv, t1[k]*(1-udat[k]), vftmp00, vftmp01; eps=eps, poi=poi, xtmp=xtmp0, tmpv=tmpv0)
            gemv!('T', 1.0, Q01, vftmp01, false, vftmp10)
            gemv!('T', 1.0, Lam1, vb0[k+1], false, vbtmp11)
            bexp!(P11, qv, t1[k]*udat[k], vbtmp11, vbtmp10; eps=eps, poi=poi, xtmp=xtmp1, tmpv=tmpv1)
            gemv!('N', 1.0, Q01, vbtmp10, false, vbtmp01)
            conv!(P00, qv, t1[k]*(1-udat[k]), vftmp00, vbtmp01, en00;
                scaling=exp(fw[k-1]+bw[k+1]-llf), eps=eps, poi=poi, xtmp=xtmp0, tmpv=tmpv0, vx=vx0)
            conv!(P11, qv, t1[k]*udat[k], vftmp10, vbtmp11, en11;
                scaling=exp(fw[k-1]+bw[k+1]-llf), eps=eps, poi=poi, xtmp=xtmp1, tmpv=tmpv1, vx=vx1)
            spger!(exp(fw[k-1]+bw[k+1]-llf), vftmp01, vbtmp10, 1.0, en01)

            gemv!('T', 1.0, Lam1, vf1[k-1], false, vftmp10)
            fexp!(P11, qv, t1[k]*udat[k], vftmp10, vftmp11; eps=eps, poi=poi, xtmp=xtmp1, tmpv=tmpv1)
            gemv!('T', 1.0, Q10, vftmp11, false, vftmp00)
            gemv!('T', 1.0, Lam0, vb0[k+1], false, vbtmp01)
            bexp!(P00, qv, t1[k]*(1-udat[k]), vbtmp01, vbtmp00; eps=eps, poi=poi, xtmp=xtmp0, tmpv=tmpv0)
            gemv!('N', 1.0, Q10, vbtmp00, false, vbtmp11)
            conv!(P11, qv, t1[k]*udat[k], vftmp10, vbtmp11, en11;
                scaling=exp(fw[k-1]+bw[k+1]-llf), eps=eps, poi=poi, xtmp=xtmp1, tmpv=tmpv1, vx=vx1)
            conv!(P00, qv, t1[k]*(1-udat[k]), vftmp00, vbtmp01, en00;
                scaling=exp(fw[k-1]+bw[k+1]-llf), eps=eps, poi=poi, xtmp=xtmp0, tmpv=tmpv0, vx=vx0)
            spger!(exp(fw[k-1]+bw[k+1]-llf), vftmp11, vbtmp00, 1.0, en10)
        end
    end

    return llf
end


eye(T::Type, m, n) = Matrix{T}(I, m, n)

function mstep!(env, en, en00, en01, en10, en11, Q, Q00, Q01, Q10, Q11)
    for (p,pv) = env
        s = 0.0
        n = 0.0
        tmp = seval(Q, p, env) .* en
        for v = tmp.nzval
            if v < 0.0
                s -= v
            elseif v > 0.0
                n += pv * v
            end
        end
        tmp = seval(Q00, p, env) .* en00
        for v = tmp.nzval
            if v < 0.0
                s -= v
            elseif v > 0.0
                n += pv * v
            end
        end
        tmp = seval(Q01, p, env) .* en01
        for v = tmp.nzval
            if v < 0.0
                s -= v
            elseif v > 0.0
                n += pv * v
            end
        end
        tmp = seval(Q10, p, env) .* en10
        for v = tmp.nzval
            if v < 0.0
                s -= v
            elseif v > 0.0
                n += pv * v
            end
        end
        tmp = seval(Q11, p, env) .* en11
        for v = tmp.nzval
            if v < 0.0
                s -= v
            elseif v > 0.0
                n += pv * v
            end
        end
        env[p] = n / s
    end
end


function makeQ(v0, B0, A0, A1, A2, K)
    Q = spzeros(AbstractMatrix{Float64}, K+1, K+1);
    @origin (Q=>0) begin
        Q[0,0] = B0
        Q[0,1] = A0
        for i = 1:K-1
            Q[i,i-1] = A2
            Q[i,i] = A1
            Q[i,i+1] = A0
        end
        Q[K,K-1] = A2
        Q[K,K] = A0 + A1
    end
    Q = sparse(block(Q))
    
    n, _ = size(B0)
    Q00 = Q[1:n,1:n]
    Q01 = Q[1:n, n .+ (1:n*K)]
    Q10 = Q[n .+ (1:n*K), 1:n]
    Q11 = Q[n .+ (1:n*K), n .+ (1:n*K)]

    Lam0 = spzeros(n*(K+1), n)
    for i = 1:n
        Lam0[i,i] = 1.0
    end
    Lam1 = spzeros(n*(K+1), n*K)
    for i = 1:n*K
        Lam1[i+n,i] = 1.0
    end
    
    alpha = [v0..., zeros(size(A1)[1]*K)...]
    xi = ones(size(Q)[1])

#    return alpha, SparseCSC(Q), SparseCSC(Q00), SparseCSC(Q01), SparseCSC(Q10), SparseCSC(Q11), SparseCSC(Lam0), SparseCSC(Lam1), xi
    return alpha, Q, Q00, Q01, Q10, Q11, Lam0, Lam1, xi
end

function mstep(param, Q, Q00, Q01, Q10, Q11, en00, en01, en10, en11)
    Q.nzval == -1
    return 1
end

function emest!(m, env, data, t00, t11; maxiter=1000, atol=1.0e-3, verbose=true, rtol=1.0e-6, idle=:idle, eps=1.0e-8, ufactor=1.01)
    params = [x for x = keys(env)]
    nn, _ = size(m.Q)
    n0 = [seval(x, env) == 1 for x = m.reward[idle]]
    n1 = [seval(x, env) != 1 for x = m.reward[idle]]
    sQ = m.Q
    sQ00 = m.Q[n0,n0]
    sQ01 = m.Q[n0,n1]
    sQ10 = m.Q[n1,n0]
    sQ11 = m.Q[n1,n1]
    salpha = m.initv
    im = sparse(eye(Float64, nn, nn))
    Lam0 = im[1:nn,n0]
    Lam1 = im[1:nn,n1];
    
    Q = seval(sQ, env)
    Q00 = seval(sQ00, env)
    Q01 = seval(sQ01, env)
    Q10 = seval(sQ10, env)
    Q11 = seval(sQ11, env)
    alpha = seval(salpha, env)
    xi = ones(nn)

    eb = similar(alpha)
    en = copy(Q)
    en00 = copy(Q00)
    en01 = copy(Q01)
    en10 = copy(Q10)
    en11 = copy(Q11)

    K = length(data)
    t0 = [t00 for _ = data]
    t1 = [t11 for _ = data]
    vf0 = Vector{Vector{Float64}}(undef, K+2)
    vf1 = Vector{Vector{Float64}}(undef, K+2)
    vb0 = Vector{Vector{Float64}}(undef, K+2)
    vb1 = Vector{Vector{Float64}}(undef, K+2)
    for i = 1:K+2
        vf0[i] = zeros(length(alpha))
        vf1[i] = zeros(length(alpha))
        vb0[i] = zeros(length(alpha))
        vb1[i] = zeros(length(alpha))
    end
    fw = zeros(K+2)
    bw = zeros(K+2)

    prev = estep!(alpha, Q, Q00, Q01, Q10, Q11, Lam0, Lam1, xi, n0, n1,
        t0, t1, data, vf0, vf1, vb0, vb1, fw, bw, eb, en, en00, en01, en10, en11; eps=eps, ufactor=ufactor)
    mstep!(env, en, en00, en01, en10, en11, sQ, sQ00, sQ01, sQ10, sQ11)
    
    iter = 1
    conv = false
    llf = 0.0
    aerror = 0.0
    rerror = 0.0
    while true
        Q = seval(sQ, env)
        Q00 = seval(sQ00, env)
        Q01 = seval(sQ01, env)
        Q10 = seval(sQ10, env)
        Q11 = seval(sQ11, env)
        alpha = seval(salpha, env)
        llf = estep!(alpha, Q, Q00, Q01, Q10, Q11, Lam0, Lam1, xi, n0, n1,
            t0, t1, data, vf0, vf1, vb0, vb1, fw, bw, eb, en, en00, en01, en10, en11; eps=eps, ufactor=ufactor)
        mstep!(env, en, en00, en01, en10, en11, sQ, sQ00, sQ01, sQ10, sQ11)
        iter += 1
        if iter >= maxiter
            println("Maximum iteration")
            break
        end
        aerror = llf - prev
        if aerror < 0.0
            println("Warning: LLF decreases at iter = $iter")
        end
        rerror = abs(aerror / prev)
        if abs(aerror) < atol && rerror < rtol
            conv = true
            break
        end
        
        if verbose
            println("iter=$iter llf=$llf aerror=$aerror rerror=$rerror")
        end
        prev = llf
    end
    (params=env, llf=llf, conv=conv, iter=iter, aerror=aerror, rerror=rerror)
end

"""
simulation
"""

function sim(rng, model, env, t)
    Q = seval(model.Q, env)
    initv = seval(model.initv, env)
    rwd = Dict([k=>seval(v, env) for (k,v) = model.reward])
    
    P, qv = NMarkov.unif(Q)
    expdist = Exponential(qv)

    state = findnext(rand(rng, Multinomial(1, initv)).==1, 1)
    simtime = 0.0
    time = [simtime]
    itime = Float64[]
    states = [state]
    rewards = Dict{Symbol,Vector{Float64}}()
    for (k,v) = rwd
        rewards[k] = [v[state]]
    end
    while simtime < t
        v = P[state,:]
        state = v.nzind[rand(rng, Multinomial(1, v.nzval)).==1][1]
        tint = rand(rng, expdist)
        simtime += tint
        push!(states, state)
        push!(itime, tint)
        push!(time, simtime)
        for (k,v) = rwd
            push!(rewards[k], v[state])
        end
    end
    (time=time, itime=itime, states=states, rewards=rewards)
end


function nexttime!(t0, alts, i)
    while true
        if i >= length(alts)
            return nothing
        end
        if t0 < alts[i]
            alts[i] -= t0
            return i
        else
            t0 -= alts[i]
            i += 1
        end
    end
end

function getbusytime!(t0, alts, r, i)
    result = 0.0
    while true
        if i >= length(alts)
            return nothing, 0.0
        end
        if t0 < alts[i]
            alts[i] -= t0
            if r[i] == 1.0
                result += t0
            end
            return i, result
        else
            t0 -= alts[i]
            if r[i] == 1.0
                result += alts[i]
            end
        end
    end
end

function get_utilization(tu, to, results)
    ## reduction (only extract the points at which the reward changes)
    prev = results.rewards[:busy][1]
    rr = [prev]
    tt = [results.time[1]]
    for i = 2:length(results.time)
        if results.rewards[:busy][i] != prev
            prev = results.rewards[:busy][i]
            push!(tt, results.time[i])
            push!(rr, prev)
        end
    end
    tt = diff(tt)
    rr = rr[1:end-1]

    ## compute utilization
    i = 1
    udat = Float64[]
    while true
        i = nexttime!(tu, tt, i)
        if i == nothing
            break
        end
        i, u = getbusytime!(to, tt, rr, i)
        if i == nothing
            break
        end
        push!(udat, u/to)
    end
    
    udat
end
