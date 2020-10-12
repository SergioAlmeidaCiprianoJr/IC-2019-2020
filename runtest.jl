using CUTEst, Printf, TimerOutputs

include("./NewtonCG/NewtonCG.jl")
include("./Cholesky/NewtonCholesky.jl")

function printheader(algorithm::Array)
    @printf("%s with %s.\n\n", algorithm[1], algorithm[3])
    @printf("fcnt:    how many times the function has been evaluated.\n")
    @printf("gcnt:    how many times the gradient has been evaluated.\n")
    @printf("hcnt:    how many times the hessian has been evaluated.\n")
    @printf("it:      problem iterations.\n")
    @printf("itSUB:   subproblem(linear solver) iterations.\n")
    @printf("itLS:    line search iterations.\n")
    @printf("time:    total run time in seconds.\n")
    @printf("timeSUB: subproblem(linear solver) run time in seconds.\n")
    @printf("timeLS:  line search run time in seconds.\n")
    @printf("SUBf:    how many times %s has failed.\n", algorithm[2])
    @printf("LSf:     how many times %s has failed.\n", algorithm[3])
    @printf("stop:    0 convergence has been achieved.\n")
    @printf("         1 maximal number of iterations exceeded.\n")
    @printf("         2 time limit exceeded.\n")

    println(repeat("_", 260))
    @printf("%-10s  %-6s  %-5s  %-15s  %-15s  %-15s  %-15s  %-15s  %-15s  %-15s  %-15s  %-15s  %-15s  %-15s  %-15s  %-15s  %-15s\n",
            "Problem", "number", "n", "f(x*)", "‖∇f(x*)‖",
            "fcnt", "gcnt", "hcnt", "it", "itSUB",
            "itLS", "time", "timeSUB", "timeBLS", "SUBf", "LSf", "stop")
end

function printinf(ans, nlp::AbstractNLPModel, number)
    totaltime = TimerOutputs.time(ans[13]["newton_modified"])/1e9
    timeSUB = timeLS = 0
    if ans[7]!=0 # it!=0
        timeSUB = TimerOutputs.time(ans[13]["newton_modified"]["linear_solver"])/1e9
        timeLS = TimerOutputs.time(ans[13]["newton_modified"]["backtrack_line_search"])/1e9
    end

    @printf("%-10s  %-6s  %-5d  %-15e  %-15e  %-15d  %-15d  %-15d  %-15d  %-15d  %-15d  %-15.5f  %-15.5f  %-15.5f  %-15d  %-15d  %-15d\n",
            nlp.meta.name, number, nlp.meta.nvar, ans[2], ans[3], ans[4],
            ans[5], ans[6], ans[7], ans[8], ans[9],
            totaltime, timeSUB, timeLS, ans[10], ans[11], ans[12])
end

function runcutest()
    io = open("CUTEst/mgh_problems", "r")
    algorithm = ["NewtonCholesky", "Cholesky", "backtrack line search"]
    #algorithm = ["NewtonCG", "Conjugate Gradients", "backtrack line search"]
    printheader(algorithm)
    for i = 1:36
        in = split(readline(io))
        if i == 1
            continue
        end
        problem = in[1]
        number = in[2]
        nlp = CUTEstModel(problem)
        algorithm[1] == "NewtonCG" ? ans = newtoncg(nlp) :
                                     ans = newtoncholesky(nlp)
        printinf(ans, nlp, number)
        finalize(nlp)
    end
    println(repeat("‾", 260))
    close(io)
end

runcutest()

function runbig()
    out = open("Tests/NewtonCG/BIG.out", "w")
    problem = ["SROSENBR", "WOODS", "POWELLSG"]
    saida = ["sigla & n & VG & AF & AG & IT & ITSP & TE & CP"]
    siglas = ["EROS", "WOOD", "EPSF"]
    EROS_N = [10, 500, 10000]
    WOOD_N = [4, 100, 10000]
    EPSF_N = [4, 20, 80, 10000]

    function especificproblem(problem)
        nlp = CUTEstModel(problem)
        ans = newtoncg(nlp)
        totaltime = round(TimerOutputs.time(ans[13]["newton_modified"])/1e9; digits=5)
        println("$(siglas[1]) & $(EROS_N[1]) & $(round(ans[2]; digits=6)) & $(ans[4]) & $(ans[5]) & $(ans[7]) & $(ans[8]) & $(totaltime) & $(ans[12]) \\\\ \\hline")

        finalize(nlp)
    end
    especificproblem(problem[1])

end

#runbig()