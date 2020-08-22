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
            ξ < H[i,j] && H[i,j]!=0 ? ξ = H[i,j] : nothing
        end
    end
    ν = maximum([1, sqrt(n^2-1)])
    ßsquared = maximum([γ, ξ/ν, eps()])
    δ = 1e-8

    for j = 1:n
        # Find the maximum diagonal value
        cmax = q = 0
        for i = j:n
            if abs(c[i][i]) > cmax
                q = i
                cmax = abs(c[i][i])
            end
        end
        # Perform row and column interchanges
        perm[j] = q
        perm[q] = j
        

        # Computes the j-th row of L
        for s = 1:j-1
            l[j][s] = c[perm[j]][s]/d[s]
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
        d[j] = maximum([δ, abs(c[perm[j]][perm[j]]), θ^2/ßsquared])

        E[j] = d[j] - c[perm[j]][perm[j]]

        # Update the prospective diagonal elements
        # and the column index
        for i = j+1:n
            permi, permj = pos(perm[i], perm[j])
            c[perm[i]][perm[i]] -= c[permi][permj]^2/d[j]
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
            xj += x[j] * (direction==1 ? l[i][j] : l[j][i])
        end
        x[i] = b[i] - xj
        j=i
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