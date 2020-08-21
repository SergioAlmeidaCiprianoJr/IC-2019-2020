using CUTEst
using Printf
using TimerOutputs

include("./NewtonCG/NewtonCG.jl")
include("./Cholesky/NewtonCholesky.jl")

function printprobleminfo(nlp::AbstractNLPModel, info::Array, output::Array, file)
    @printf(file, "CUTEst problem................: %s\n", info[1])
    @printf(file, "Number of variables...........: %d\n", nlp.meta.nvar)
    @printf(file, "Maximum problem iterations....: %d\n", info[2])
    @printf(file, "Maximum subproblem iterations.: %d\n", info[3])
    @printf(file, "Maximum line search iterations: %d\n", info[4])
    @printf(file, "Time limit....................: %s\n\n", info[5])
    
    totaltime = TimerOutputs.time(output[13]["newton_modified"])/1e9
    timeSUB = timeLS = 0
    if output[7]!=0 # it!=0
        timeSUB = TimerOutputs.time(
            output[13]["newton_modified"]["linear_solver"])/1e9
        timeLS = TimerOutputs.time(
            output[13]["newton_modified"]["backtrack_line_search"])/1e9
    end

    @printf(file, "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ summary statistics ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n")
    @printf(file, "Objective.............: %e\n", output[2])
    @printf(file, "Gradient norm.........: %e\n", output[3])
    @printf(file, "Total iterations......: %d\n", output[7])
    @printf(file, "Total time in seconds.: %f\n", totaltime)
    @printf(file, "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n\n")

    @printf(file, "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ output statistics ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n")
    @printf(file, "Number of objective function evaluations: %d\n", output[4])
    @printf(file, "Number of gradient evaluations..........: %d\n", output[5])
    @printf(file, "Number of hessian evaluations...........: %d\n", output[6])
    @printf(file, "Subproblem iterations...................: %d\n", output[8])
    @printf(file, "Subproblem time in seconds..............: %f\n", timeSUB)
    @printf(file, "How many times subproblem failed........: %d\n", output[10])
    @printf(file, "Line search iterations..................: %d\n", output[9])
    @printf(file, "Line search time in seconds.............: %f\n", timeLS)
    @printf(file, "How many times line search failed.......: %d\n", output[11])
    @printf(file, "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n\n")    
    
    @printf(file, "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ superscription ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n")
    @printf(file, "iter....: current iteration\n")
    @printf(file, "f(x*)...: objective function avaluated at x(iter)\n")
    @printf(file, "‖∇f(x)‖.: gradient norm used to calculate x(iter), so x in ‖∇f(x)‖ is equal to x(iter-1)\n")
    @printf(file, "alpha...: step calculated by backtrack line search\n")
    @printf(file, "‖d‖.....: search direction norm\n")
    @printf(file, "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n")

    println(file, repeat("_", 74))
    @printf(file, "%-6s  %-15s  %-15s  %-15s  %-15s\n",
            "iter", "f(x*)", "‖∇f(x)‖", "alpha", "‖d‖")
    allobj = output[14][1]
    all∇f = output[14][2]
    allalpha = output[14][3]
    allpnorm = output[14][4]
    for i = 1:output[7]
        @printf(file, "%-6d  %-15e  %-15e  %-15e  %-15e\n",
                i, allobj[i], norm(all∇f[i]), allalpha[i], allpnorm[i])
    end
    println(file, repeat("‾", 74))
    println(file, output[13])

    if(output[12] == 0)
        EXIT = "convergence has been achieved."
    elseif(output[12] == 1)
        EXIT = "maximal number of iterations exceeded."
    else
        EXIT = "time limit exceeded."
    end
    println(file, "\nEXIT: $(EXIT)\n")
end

norm(f) = sqrt(sum(f.*f))

function runseparate(algorithm, file, nproblem)
    # file with all the input
    io = open("CUTEst/$(file)", "r")

    info = [" ", 1000, 1000, 1000, "10 min"]
    # info[1] -> CUTEst problem
    # info[2] -> Maximum problem iterations
    # info[3] -> Maximum subproblem iterations
    # info[4] -> Maximum line search iterations
    # info[5] -> Time limit

    # skip first line (it is a comment)
    split(readline(io))
    name = "empty"
    number = n = m = 0

    while nproblem != number
        name, number, n, m = split(readline(io))
        number = parse(Int64, number)
        n = parse(Int64, n)
        m = parse(Int64, m)
    end

    info[1] = name
    println("Started $(info[1])")
    
    nlp = CUTEstModel(info[1])
    if algorithm == "NewtonCG"
        output = newtoncg(nlp)
    else
        output = newtoncholesky(nlp)
        # subproblem iterations depend on n
        info[3] = nlp.meta.nvar
    end

    out = open("Testes/$(algorithm)/$(file)/$(info[1]).out", "w")
    println(out, "$(algorithm) with backtrack line search.\n")
    printprobleminfo(nlp, info, output, out)

    println("Finished $(info[1])")
    # finalizing all processes
    finalize(nlp)
    close(out)
    close(io)
end

#algorithm = "NewtonCG"
 algorithm = "NewtonCholesky"

 file = "mgh_problems"
#file = "mgh_ne"
#file = "mgh_nls"
#file = "mgh_unmin"

#nproblem = readline()
#nproblem = parse(Int64, nproblem)

for nproblem = 1:35
    runseparate(algorithm, file, nproblem)
end
