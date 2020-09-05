using NLPModels

function solvelinear(H, ∇f::Array)
    # Considering a system Ax = b,
    # where A = H, x = p = goal, b = -∇f          
    
    # Cholesky factorization
    l, d, n, perm = ldl(H)

    # b = P b
    p = b = zeros(Float64, n)
    b[:] = -∇f[perm[:]]

    # L x = b, b = -∇f
    x = solvelowerl(l, b, n)

    # D x* = x  ->  x* = D^-1 x
    for i=1:n
        x[i] /= d[i]
    end
    
    # L^T x** = b, b = x*
    x = solveupperl(l, x, n)

    # p = P^T x
    p[perm[:]] = x[:]

    # return search direction and total iterations
    return p, n*(n/2) + 2*n + n
end

function ldl(H)
    # this function computes the Cholesky factorization of
    # H + E = L D L^T

    # outputs
    n = size(H,1)
    l = zeros(Float64, trunc(Int, n*(n-1)/2))
    d = zeros(Float64, n)
    perm = [i for i = 1:n]

    # local arrays
    c = zeros(Float64, trunc(Int, n*(n-1)/2))
    cd = zeros(Float64, n)

    # initialize constants
    γ = H[1,1]
    ξ = H[1,n]
    for i = 1:n, j = 1:n
        if i == j
            cd[i] = H[i,i]
            # Find the maximum diagonal value
            γ < H[i,i] ? γ = H[i,i] : nothing
        else
            # Find the maximum off-diagonal value
            ξ < H[i,j] ? ξ = H[i,j] : nothing
        end
    end
    ν = maximum([1, sqrt(n^2-1)])
    ß2 = maximum([γ, ξ/ν, eps()])
    δ = 1e-8

    for j = 1:n
        # Find the maximum diagonal value
        cmax = cd[perm[j]]
        q = j
        for i = j+1:n
            if abs(cd[perm[i]]) > cmax
                q = i
                cmax = abs(cd[perm[i]])
            end
        end
        # Perform row and column interchanges
        temp = perm[j]
        perm[j] = perm[q]
        perm[q] = temp

        # Computes the j-th row of L
        for s = 1:j-1
            l[pos(j, s)] = c[pos(perm[j], perm[s])] / d[s]
        end

        # Find the maximum modulus of lij * dj
        θ = 0
        for i = j+1:n
            sum = 0
            for s = 1:j-1
                sum += l[pos(j, s)] * c[pos(perm[i], perm[s])]
            end

            cpos = pos(perm[i], perm[j])
            pi, pj = igtj(perm[i], perm[j])

            c[cpos] = H[pi, pj] - sum
            θ = maximum([c[cpos], θ])
        end

        # Compute the j-th diagonal element of D
        d[j] = maximum([δ, abs(cd[perm[j]]), θ^2/ß2])

        # Update the prospective diagonal elements
        # and the column index
        for i = j+1:n
            cd[perm[i]] -= c[pos(perm[i], perm[j])]^2/d[j]
        end
    end

    return l, d, n, perm
end

function solvelowerl(l::Array{Float64,1}, b::Array, n::Integer)
    # solve Lx = b and returns x
    # for L being a triangular matrix with unit diagonal
    x = zeros(Float64, n)
    for i = 1:n
        xj = 0
        for j = 1:i-1
            xj += x[j] * l[pos(i, j)]
        end
        x[i] = b[i] - xj
    end
    return x
end

function solveupperl(l::Array{Float64,1}, b::Array, n::Integer)
    # solve Lx = b and returns x
    # for L being a triangular matrix with unit diagonal
    x = zeros(Float64, n)
    for i = n:-1:1
        xj = 0
        for j = i+1:n
            xj += x[j] * l[pos(j, i)]
        end
        x[i] = b[i] - xj
    end
    return x
end

function pos(i, j)
    i, j = igtj(i, j)
    return trunc(Int, (i-1) * (i-2) / 2 + j)
end

function igtj(i, j)
    i > j ? (return i, j) : (return j, i)
end
