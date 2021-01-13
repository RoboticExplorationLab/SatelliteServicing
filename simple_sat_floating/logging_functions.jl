"""This is all for inner/outer loop logging"""

function solver_logging(iter,DJ,l,J,α)
    """Don't worry about this, just for visual logging in the REPL"""
    if rem((iter-1),4)==0
    printstyled("iter     α              maxL           J              DJ\n";
                    bold = true, color = :light_blue)
        bars = "----------------------------"
        bars*="------------------------------------\n"
        printstyled(bars; color = :light_blue)
    end
    DJ = @sprintf "%.4E" DJ
    maxL = @sprintf "%.4E" round(maximum(maximum.(l)),sigdigits = 3)
    J_display = @sprintf "%.4E" J
    alpha_display = @sprintf "%.4E" α
    str = "$iter"
    for i = 1:(6 - ndigits(iter))
        str *= " "
    end
    println("$str   $alpha_display     $maxL     $J_display     $DJ")
    return nothing
end
function outer_loop_solver_logging(iter,total,c_max,μ,ϕ)
    """Don't worry about this, just for visual logging in the REPL"""
    printstyled("iter     total          c_max          μ              ϕ\n";
                    bold = true, color = :light_magenta)
        bars = "----------------------------"
        bars*="------------------------------------\n"
        printstyled(bars; color = :light_magenta)
    c_max = @sprintf "%.4E" c_max
    μ = @sprintf "%.4E" μ
    ϕ = @sprintf "%.4E" ϕ
    str = "$iter"
    for i = 1:(6 - ndigits(iter))
        str *= " "
    end
    str_total = "$total"
    for i = 1:(6 - ndigits(total))
        str_total *= " "
    end
    println("$str   $str_total         $c_max     $μ     $ϕ")
    return nothing
end
