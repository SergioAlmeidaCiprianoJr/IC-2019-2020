using NLPModels

function backtracklinesearch(x::Array, nlp::AbstractNLPModel, p::Array, ∇f::Array)
	# output: alpha, iterations, failure
	alpha = 1
	c = 1e-4
	p_lo = 0.1
	p_hi = 0.5
	i = 0
	limit = 1000
	failure = 0

	∇fTp = sum(∇f.*p)
	objx = obj(nlp, x)
	while obj(nlp, x + alpha.*p) > objx + c*alpha*∇fTp
		alpha = min(alpha, alpha*p_hi) # avoid too big reductions
		alpha = max(alpha, alpha*p_lo) # avoid too small reductions

		i+=1
		if i>=limit
			failure = 1
			break
		end
	end

	return alpha, i, failure
end
