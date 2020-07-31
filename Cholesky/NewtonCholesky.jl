using NLPModels, TimerOutputs
import LinearOperators.LinearOperator
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
            
            H = hess(nlp, x)
            p, j = @timeit to "linear_solver" solvelinear(H, ∇f)
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

function solvelinear(H::LinearOperator, ∇f::Array)
    # output: search direction, iterations
    l, d, n = ldl(H)
    it = n*(n/2) + 2*n + n # ldl + 2*solveforl + solveDiagonal
    # Lz = b, b = -∇f
    z = solveforl(l, -∇f, n, 1)
    # Dz* = z -> z* = D^-1z
    [ z[i] = z[i]*(1/d[i]) for i=1:n ]
    # L^Tx = z*
    return solveforl(l, z, n, -1), it
end

function ldl(H::LinearOperator)
    # output: lower triangular matrix, diagonal (matrix) vector, hessian nrows
    n = size(H,1)
    d = zeros(Float64, n)
    c = [zeros(_) for _ = 1:n]
    l = [zeros(_) for _ = 1:n]
    ß = 1e-3
    Δ = 1e-8
    max = 0
    for j = 1:n
        d[j] = maximum([abs(c[j][j]), (max/ß)^2, Δ])
        c[j][j] = H[j, j] - sumto(j, j, d, l, false)
        max = 0
        # l[j][j] = 1
        for i = j+1:n
            c[i][j] = H[i, j] - sumto(i, j, d, l, false)
            abs(c[i][j]) > max ? max = abs(c[i][j]) : nothing
            l[i][j] = c[i][j]/d[j]
        end
    end
    return l, d, n
end

function solveforl(l::Array{Array{Float64,1},1}, b::Array, n::Integer, direction::Integer)
    # solve Lx = b and returns x
    # for L being a triangular matrix with unit diagonal
    x = zeros(Float64, n)
    j = 1; k = n
    if direction == -1 
        j = n; k = 1
    end
    for i = j:direction:k
        x[i] = b[i] - sumto(0, i, x, l, true)
    end
    return x
end

function sumto(i::Integer, j::Integer, d::Array, l::Array, key::Bool)
    # sum x*l
    # and all sums l*d*l
    dl = 0.0
    for s = 1:j-1
        key ? dl += d[s] * l[j][s] :
              dl += d[s] * l[j][s] * l[i][s]
    end
    return dl
end