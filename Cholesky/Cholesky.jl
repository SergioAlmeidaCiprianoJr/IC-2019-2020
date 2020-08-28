using NLPModels

function solvelinear(H, ∇f::Array)
    # Considering a system Ax = b,
    # where A = H, x = p = goal, b = -∇f          
    
    # Cholesky factorization
    l, d, n, perm = ldl(H)

    # b = P b
    b = zeros(Float64, n)
    b[:] = -∇f[perm[:]]

    # L x = b, b = -∇f
    x = solvelowerl(l, b, n)

    # D x* = x -> x* = D^-1 x
    for i=1:n
        x[i] /= d[i]
    end
    
    # L^T x** = b, b = x*
    x = solveupperl(l, x, n)

    # p = P^T x**
    p = zeros(Float64, n)
    p[perm[:]] = x[:]

    # return search direction and total iterations
    return p, n*(n/2) + 2*n + n
end

function ldl(H)
    # this function computes the Cholesky factorization of
    # H + E = L D L^T

    # outputs
    n = size(H,1)
    l = [zeros(_) for _ = 1:n]
    d = zeros(Float64, n)

    # local variables
    E = zeros(Float64, n)
    perm = [i for i = 1:n]
    c = [zeros(_) for _ = 1:n]

    # initialize constants
    γ = H[1,1] 
    ξ = H[1,n]
    for i = 1:n, j = 1:n
        if i == j
            c[i][i] = H[i,i]
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
        cmax = c[perm[j]][perm[j]]
        q = temp = perm[j]
        for i = j+1:n
            if abs(c[perm[i]][perm[i]]) > cmax
                q = perm[i]
                cmax = abs(c[q][q])
            end
        end
        # Perform row and column interchanges
        perm[j] = perm[q]
        perm[q] = temp

        # Computes the j-th row of L
        for s = 1:j-1
            permj, perms = pos(perm[j], s)
            l[j][s] = c[permj][perms]/d[s]
        end

        # Find the maximum modulus of lij * dj
        θ = 0
        for i = j+1:n
            sum = 0
            for s = 1:j-1
                permi, perms = pos(perm[i], s)
                sum += l[j][s]*c[permi][perms]
            end

            permi, permj = pos(perm[i], perm[j])
            c[permi][permj] = H[permi, permj] - sum
            cpos = c[permi][permj]
            θ < abs(cpos) ? θ = abs(cpos) : nothing
        end

        # Compute the j-th diagonal element of D
        d[j] = maximum([δ, abs(c[perm[j]][perm[j]]), θ^2/ß2])

        E[j] = d[j] - c[perm[j]][perm[j]]

        # Update the prospective diagonal elements
        # and the column index
        for i = j+1:n
            permi, permj = pos(perm[i], perm[j])
            c[perm[i]][perm[i]] -= c[permi][permj]^2/d[j]
        end
    end

    return l, d, n, perm, E
end

function solvelowerl(l::Array{Array{Float64,1},1}, b::Array, n::Integer)
    # solve Lx = b and returns x
    # for L being a triangular matrix with unit diagonal
    x = zeros(Float64, n)
    for i = 1:n
        xj = 0
        for j = 1:i-1
            xj += x[j] * l[i][j]
        end
        x[i] = b[i] - xj
    end
    return x
end

function solveupperl(l::Array{Array{Float64,1},1}, b::Array, n::Integer)
    # solve Lx = b and returns x
    # for L being a triangular matrix with unit diagonal
    x = zeros(Float64, n)
    for i = n:-1:1
        xj = 0
        for j = i+1:n
            xj += x[j] * l[j][i]
        end
        x[i] = b[i] - xj
    end
    return x
end

function pos(i, j)
    i > j ? (return i, j) : (return j, i)
end

function ldlproduct(n, lin, din, Ein, H)
    # this function computes ldlt to test
    # whether ldl function worked
    l = zeros(n, n)
    d = zeros(n, n)
    E = zeros(n, n)

    for j = 1:n
        for i = j:n
            l[i,j] = lin[i][j]
        end
    end

    for i = 1:n
        d[i,i] = din[i]
        E[i,i] = Ein[i]
        l[i,i] = 1
    end

    lt = transpose(l)
    return l*d*lt, H+E
end