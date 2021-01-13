using LinearAlgebra, ForwardDiff

function c_fx(x,u)
    """Constraint function, c_fx(x,u)<0 means constraint is satisfied"""
    return [u .- 5 ;
           -u .+ -5]
end
function Π(z)
    """Projection operator that projects onto the positive numbers"""
    out = zeros(eltype(z),length(z))
    for i = 1:length(out)
        out[i] = min(0,z[i])
    end
    return out
end
function Lag(Q,R,x,u,xf,λ,μ)
    """Augmented Lagrangian"""
    return (LQR_cost(Q,R,x,u,xf) +
            (1/(2*μ))*(  norm(Π(λ - μ*c_fx(x,u)))^2 - dot(λ,λ)))
end
function I_mu(u,λ,μ)
    
Q = Diagonal(randn(18))
R = Diagonal(randn(9))

#cx'*Iμ*cx
λ = randn(18)
μ = 13.4
x = randn(18)
xf = randn(18)
u = 5*randn(9)
L_fxu(ul) = Lag(Q,R,x,ul,xf,λ,μ)

# ∂²J/∂u²
R = FD.hessian(L_fxu,u)

# ∂J/∂u
r = FD.gradient(L_fxu,u)


cx = [I(9);-I(9)]

cx_fx(ul) = c_fx(x,ul)

@show norm(cx -FD.jacobian(cx_fx,u))
