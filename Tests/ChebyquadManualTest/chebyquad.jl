function T(i, x)
    # Calculando o i-ésimo polinômio de Chebyshev
    # Baseado em https://www-m2.ma.tum.de/foswiki/pub/M2/Allgemeines/SemWs09/testprob.pdf
    if i == 0
        return 1
    elseif i == 1
        return x
    end
    return 2*x*T(i-1, x) - T(i-2, x)
end

n = 10
m = 10
x = [0.0124447, 0.149118, 0.22936,
     0.35929, 0.445338, 0.569651, 0.627111,
     0.798093, 0.856879, 1.00918]
f = zeros(Float64, n)

for i=1:m
    integral = i%2 != 0 ? 0 : -1/(i^2-1)

    somaTi = 0
    for j=1:n
        somaTi += T(i, x[j])
    end

    f[i] = 1/n * somaTi - integral
    println("f[$(i)] = $(f[i])")
end

chebyqad = 0
for i=1:m
    global chebyqad += f[i]^2
end
println("chebyqad = $(chebyqad)")