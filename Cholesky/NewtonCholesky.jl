using NLPModels, TimerOutputs
include("../LineSearch/backtrack_line_search.jl")
include("./Cholesky.jl")

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
			∇fnorm = sqrt(sum(∇f.*∇f))
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
	return [x, obj(nlp, x), ∇fnorm, fcnt, gcnt, hcnt, it, itSUB, itBLS, 0, BLSf, stop, to, values]
end
