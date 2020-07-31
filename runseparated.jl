using CUTEst, Printf, TimerOutputs

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

    @printf(file, "Total iterations........................: %d\n", output[7])
    @printf(file, "Subproblem iterations...................: %d\n", output[8])
    @printf(file, "Line search iterations..................: %d\n", output[9])
    @printf(file, "Number of objective function evaluations: %d\n", output[4])
    @printf(file, "Number of gradient evaluations..........: %d\n", output[5])
    @printf(file, "Number of hessian evaluations...........: %d\n", output[6])
    @printf(file, "Total time in seconds...................: %f\n", totaltime)
    @printf(file, "Subproblem time in seconds..............: %f\n", timeSUB)
    @printf(file, "Line search time in seconds.............: %f\n", timeLS)
    @printf(file, "How many times subproblem failed........: %d\n", output[10])
    @printf(file, "How many times line search failed.......: %d\n\n", output[11])

    println(file, repeat("_", 75))
    @printf(file, "%-6s  %-15s  %-15s  %-15s  %-15s\n",
            "iter", "f(x*)", "‖∇f(x*)‖", "alpha", "‖d‖")
    allobj = output[14][1]
    all∇f = output[14][2]
    allalpha = output[14][3]
    allpnorm = output[14][4]
    for i = 1:output[7]
        @printf(file, "%-6d  %-15e  %-15e  %-15e  %-15e\n",
                i, allobj[i], norm(all∇f[i]), allalpha[i], allpnorm[i])
    end
    println(file, repeat("‾", 75))

    @printf(file, "\nObjective.............: %e\n", output[2])
    @printf(file, "Gradient norm.........: %e\n", output[3])
    @printf(file, "Time..................: %e\n", totaltime)
end

norm(f) = sqrt(sum(f.*f))

function runcutest()
    io = open("CUTEst/cp", "r")

    #algorithm = "NewtonCG"
    algorithm = "NewtonCholesky"
    info = [" ", 1000, 1000, 1000, "10 min"]

    for i = 1:80
        in = split(readline(io))
        info[1] = in[1]
        info[1] == "CHEBYQAD" ? nlp = CUTEstModel("CHEBYQAD", "-param", "N=10") :
                                nlp = CUTEstModel(info[1])
        if(algorithm == "NewtonCG")
            output = newtoncg(nlp)
            file = open("Testes/NewtonCG/$(info[1]).out", "w")
            @printf(file, "NewtonCG with backtrack line search.\n\n")
        else
            output = newtoncholesky(nlp)
            file = open("Testes/NewtonCholesky/$(info[1]).out", "w")
            @printf(file, "NewtonCholesky with backtrack line search.\n\n")
            info[3] = nlp.meta.nvar
        end
        
        if(output[12] == 0)
            EXIT = "convergence has been achieved."
        elseif(output[12] == 1)
            EXIT = "maximal number of iterations exceeded."
        else
            EXIT = "time limit exceeded."
        end
        
        printprobleminfo(nlp, info, output, file)
        @printf(file, "EXIT: %s\n", EXIT)
        finalize(nlp)
        close(file)
        break
    end
    close(io)
end

runcutest()