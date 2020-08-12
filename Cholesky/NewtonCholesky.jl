using NLPModels, TimerOutputs
include("../LineSearch/backtrack_line_search.jl")

# Newton-Cholesky with backtrack line search.
# outputs
# fcnt:    how many times the function has been evaluated.
# gcnt:    how many times the gradient has been evaluated.
# hcnt:    how many times the hessian has been evaluated.
# it:      problem iterations.
# itSUB:   subproblem(linear solver) iterations.
# itLS:    line search iterations.
# time:    total run time.
# timeSUB: subproblem(linear solver) run time.
# timeLS:  line search run time.
# SUBf:    how many times linear solver has failed.
# LSf:     how many times line search has failed.
# stop:    0 convergence has been achieved.
#          1 Newton-Cholesky failed to converge.
#          2 time limit exceeded.

function newtoncholesky(nlp::AbstractNLPModel; tle = 10, e = 1e-8, itMAX = 1e3)
    to = TimerOutput()
    @timeit to "newton_modified" begin
        # output variables
        fcnt = gcnt = hcnt = 0
        it = itSUB = itBLS = 0
        stop = 1
        BLSf = 0
		allobj = zeros(Float64, Integer(itMAX))
		all∇f = zeros(Float64, Integer(itMAX))
		allalpha = zeros(Float64, Integer(itMAX))
		allpnorm = zeros(Float64, Integer(itMAX))

        ∇f = ∇fnorm = 0
        x = nlp.meta.x0
        while it<itMAX   
            ∇f = grad(nlp, x)
            ∇fnorm = norm(∇f)
            gcnt += 1
            if ∇fnorm < e
                stop = 0
                break
            end
            
            p, j = @timeit to "linear_solver" solvelinear(hess(nlp, x), ∇f)
            itSUB += j
            hcnt += 1
            
            alpha, i, failure = @timeit to "backtrack_line_search" backtracklinesearch(x, nlp, p, ∇f)
            itBLS += i
            BLSf += failure
            
            x = x + alpha.*p
            it += 1

            # saving data
            allobj[it] = obj(nlp, x)
            fcnt+=1
            all∇f[it] = ∇fnorm
            allalpha[it] = alpha
            allpnorm[it] = norm(p)
            
            # time limit
            totaltime = (TimerOutputs.time(to["newton_modified"]["backtrack_line_search"]) +
            TimerOutputs.time(to["newton_modified"]["linear_solver"]))/1e9
            if totaltime >= tle*60 # minutes
                stop = 2
				break
            end
        end
    end
    values = [allobj, all∇f, allalpha, allpnorm]
    fcnt += itBLS
    println("x = $(x)")
	return [x, obj(nlp, x), ∇fnorm, fcnt, gcnt, hcnt, it, itSUB, itBLS, 0, BLSf, stop, to, values]
end

function solvelinear(H, ∇f::Array)
    # output: search direction, iterations
    l, d, n = ldl(H)
    it = n*(n/2) + 2*n + n # ldl + 2*solveforl + solveDiagonal iterations
    # Lx = b, b = -∇f
    x = solveforl(l, -∇f, n, 1)
    # Dx* = x -> x* = D^-1x
    # x* is the result
    for i=1:n
        x[i] /= d[i]
    end
    # L^Tx = b, b = x*
    return solveforl(l, x, n, -1), it
end

function ldl(H)
    # output: lower triangular matrix, diagonal (matrix) vector, hessian nrows
    n = size(H,1)
    d = zeros(Float64, n)
    E = zeros(Float64, n)
    c = [zeros(_) for _ = 1:n]
    l = [zeros(_) for _ = 1:n]
    γ = H[1,1] 
    ξ = H[1,2]
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
    #ßsquared = 1e-6
    δ = 1e-8

    for j = 1:n
        for s = 1:j-1
            l[j][s] = c[j][s]/d[j]
        end 
        _maxc = 0
        for i = j+1:n
            _sum = 0
            for s = 1:j-1
                _sum += l[j][s]*c[i][s]
            end
            c[i][j] = H[i][j] - _sum
            _maxc < abs(c[i][j]) ? _maxc = abs(c[i][j]) : nothing
        end
        d[j] = maximum([δ, abs(c[j][j]), _maxc^2/ßsquared])
        E[j] = d[j] - c[j][j]
        for i = j+1:n
            c[i][i] -= c[i][j]^2/d[j]
        end
    end

    return l, d, n
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
