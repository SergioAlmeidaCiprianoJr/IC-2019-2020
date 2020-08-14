using NLPModels

function solvelinear(H, ∇f::Array)
    # Considering a system Ax = b,
    # where A = H, x = goal, b = ∇f           
    
    # Cholesky factorization
    l, d, n = ldl(H)

    # Lx = b, b = -∇f
    x = solveforl(l, -∇f, n, 1)

    # Dx* = x -> x* = D^-1x
    for i=1:n
        x[i] /= d[i]
    end
    
    # L^Tx = b, b = x*
    # return search direction and total iterations
    return solveforl(l, x, n, -1), n*(n/2) + 2*n + n
end

function ldl(H)
    # this routine computes the Cholesky factorization of
    # H + E = L D L^T

    # outputs
    n = size(H,1)
    l = [zeros(_) for _ = 1:n]
    d = zeros(Float64, n)

    # local variables
    E = zeros(Float64, n)
    c = [zeros(_) for _ = 1:n]

    # constants
    γ = H[1,1] 
    ξ = H[1,n]
    for i = 1:n, j = 1:n
        if i == j
            c[i][i] = H[i,i]
            # Find the maximum diagonal value
            γ < H[i,i] ? γ = H[i,i] : nothing
        else
            # Find the maximum off-diagonal value
            ξ < H[i,j] && H[i,j]!=0 ? ξ = H[i,j] : nothing
        end
    end
    ν = maximum([1, sqrt(n^2-1)])
    ßsquared = maximum([γ, ξ/ν, eps()])
    δ = 1e-8

    for j = 1:n
        for s = 1:j-1
            l[j][s] = c[j][s]/d[s]
        end
        _maxc = 0
        for i = j+1:n
            _sum = 0
            for s = 1:j-1
                _sum += l[j][s]*c[i][s]
            end
            c[i][j] = H[i,j] - _sum
            _maxc < abs(c[i][j]) ? _maxc = abs(c[i][j]) : nothing
        end
        d[j] = maximum([δ, abs(c[j][j]), _maxc^2/ßsquared])
        E[j] = d[j] - c[j][j]
        for i = j+1:n
            c[i][i] -= c[i][j]^2/d[j]
        end
    end

    return l, d, n, E
end

function solveforl(l::Array{Array{Float64,1},1}, b::Array, n::Integer, direction::Integer)
    # solve Lx = b and returns x
    # for L being a triangular matrix with unit diagonal
    x = zeros(Float64, n)
    start = n
    finish = 1
    if direction==1
        start = 1
        finish = n
    end
    xj = j = 0
    for i = start:direction:finish
        if j>0
            xj += x[j] * ( direction==1 ? l[i][j] : l[j][i] )
        end
        x[i] = b[i] - xj
        j=i
    end
    return x
end
