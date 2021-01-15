cd(joinpath(dirname(dirname(@__FILE__)),"Julia_Environment"))
Pkg.activate(".")
Pkg.instantiate()
using ForwardDiff, LinearAlgebra, MATLAB, Infiltrator
using Attitude, StaticArrays
using Printf
using DiffResults
const FD = ForwardDiff

load in dynamics function
include(joinpath(dirname(@__FILE__),"dynamics_setup.jl"))

# load logging utilities
include(joinpath(dirname(@__FILE__),"logging_functions.jl"))

function rk4(f, x_n, u, t,dt)
    """RK4 for integration of a single time step"""
    k1 = dt*f(x_n,u,t)
    k2 = dt*f(x_n+k1/2, u,t+dt/2)
    k3 = dt*f(x_n+k2/2, u,t+dt/2)
    k4 = dt*f(x_n+k3, u,t+dt)
    return (x_n + (1/6)*(k1+2*k2+2*k3 + k4))

end

function discrete_dynamics(x,u,t,dt)
    return rk4(dynamics,x,u,t,dt)
end

function LQR_cost(Q,R,x,u,xf)
    """Standard LQR cost function with a goal state"""
    return .5*(x-xf)'*Q*(x - xf) + .5*u'*R*u
end

function c_fx(x,u)
    """Constraint function, c_fx(x,u)<0 means constraint is satisfied"""
    return [u - params.u_max;
           -u + params.u_min]
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
    # return LQR_cost(Q,R,x,u,xf)
end
function term_cost(x,xf,λ,μ)
    """AL terms for goal constraint"""
    c = (x - xf)
    return -dot(λ,c) + .5*μ*dot(c,c)
end
function al_ilqr(x0,utraj,xf,Q_lqr,Qf_lqr,R_lqr,N,dt,μ,λ)

    # problem status
    prob_success = true

    # state and control dimensions
    Nx = length(x0)
    Nu = size(R_lqr,1)

    # initial trajectory is initial conditions the whole time
    xtraj = cfill(Nx,N)
    xtraj[1] = copy(x0)

    # initial forward rollout, all we do is simulate and get the cost
    J = 0.0
    for k = 1:N-1
        xtraj[k+1] = discrete_dynamics(xtraj[k],utraj[k],(k-1)*dt,dt)
        J += Lag(Q_lqr,R_lqr,xtraj[k],utraj[k],xf,λ[k],μ)
    end
    J += .5*(xtraj[N]-xf)'*Qf_lqr*(xtraj[N] - xf)+term_cost(xtraj[N],xf,λ[N],μ)

    # allocate K and l
    K = cfill(Nu,Nx,N-1)
    l = cfill(Nu,N-1)

    # allocate the new states and controls
    xnew = cfill(Nx,N)
    unew = cfill(Nu,N-1)

    output_iter = 0
    # main loop
    for iter = 1:50

        # Cost to go hessian and gradient at final time step
        S = copy(Qf_lqr) + μ*I # added goal state constraint terms
        s = Qf_lqr*(xtraj[N] - xf) - λ[N] + μ*(xtraj[N]-xf) # " "

        # backwards pass
        for k = (N-1):-1:1

            # ∂²J/∂x²
            Q = copy(Q_lqr)

            # ∂J/∂x
            q = Q_lqr*(xtraj[k] - xf)

            # closure for control derivatives
            L_fxu(ul) = Lag(Q_lqr,R_lqr,xtraj[k],ul,xf,λ[k],μ)

            # ∂²J/∂u²
            R = FD.hessian(L_fxu,utraj[k])

            # ∂J/∂u
            r = FD.gradient(L_fxu,utraj[k])

            # discrete dynamics jacobians
            A_fx(xloc) = discrete_dynamics(xloc,utraj[k],(k-1)*dt,dt)
            Ak = FD.jacobian(A_fx,xtraj[k])
            B_fx(u_loc) =  discrete_dynamics(xtraj[k],u_loc,(k-1)*dt,dt)
            Bk = FD.jacobian(B_fx,utraj[k])

            # solve
            F = factorize(R + Bk'*S*Bk + (1e-8)*I)
            l[k] = F\(r + Bk'*s)
            K[k] = F\(Bk'*S*Ak)

            # update
            Snew = Q + K[k]'*R*K[k] + (Ak-Bk*K[k])'*S*(Ak-Bk*K[k])
            snew = q - K[k]'*r + K[k]'*R*l[k] + (Ak-Bk*K[k])'*(s - S*Bk*l[k])

            # update S's
            S = copy(Snew)
            s = copy(snew)

        end

        # initial conditions for new trajectory
        xnew[1] = copy(x0)

        # learning rate
        α = 1.0

        # forward pass (with line search)
        Jnew = 0.0
        for inner_i = 1:20
            Jnew = 0.0

            # rollout the dynamics
            for k = 1:N-1
                unew[k] = utraj[k] - α*l[k] - K[k]*(xnew[k]-xtraj[k])
                xnew[k+1] = discrete_dynamics(xnew[k],unew[k],(k-1)*dt,dt)
                Jnew += Lag(Q_lqr,R_lqr,xnew[k],unew[k],xf,λ[k],μ)
            end
            Jnew += .5*(xnew[N]-xf)'*Qf_lqr*(xnew[N] - xf)+term_cost(xnew[N],xf,λ[N],μ)

            # if the new cost is lower, we keep the new trajectory
            if Jnew<J
                break
            else# this also pulls the linesearch back if hasnan(xnew)
                α /= 2
            end

            # this will happen if linearization is bad for some reason, or
            # if the trajectory has converged and can't do any better
            if inner_i == 20
                # this causes termination of the solver since DJ = 0
                Jnew = copy(J)

                # this flag tells us to not update the states and controls
                prob_success = false
                break
            end

        end

        # update trajectory and control history
        if prob_success
            xtraj = copy(xnew)
            utraj = copy(unew)
        end

        # termination criteria
        DJ = abs(J - Jnew)
        solver_logging(iter,DJ,l,J,α)
        if DJ<params.dJ_tol
            output_iter = iter
            break
        else
            J = Jnew
        end

    end

    return xtraj, utraj, K, output_iter
end


function runit()

# constraints on u
u_min = -10*(@SVector ones(13))
u_max = 10*(@SVector ones(13))

# LQR cost function (.5*(x-xf)'*Q*(x - xf) + .5*u'*R*u)
Q = 100*Diagonal(ones(26))
Q[1:3,1:3]*=100
Qf = Q
R = Diagonal(@SVector ones(13))

# solver state
# x = [mrp; position; joint angles; ω; vel; joint speeds]

# RBD state (used purely inside the dynamics function)
# x = [quaternion; position; joint angles; ω; vel; joint speeds]

x0 = zeros(26)  # initial state
final_joint_angles = [.1;.2;-.4;.5;.8;-.3;.7]
xf = [.414;.3;.1;5;2;3;final_joint_angles;zeros(13)]

# global named tuple to pass to solver
global params = (dJ_tol = 1e-2, u_min = u_min, u_max = u_max,ϵ_c = 1e-3)

# time step size and number of time steps
dt = 0.1
N = 300

# Augmented Lagrangian stuff
μ = 1
ϕ = 5
λ = cfill(26,N)
# goal constraint λ ∈ R^26
λ[N] = zeros(26)
utraj = [0*randn(13) for i = 1:N-1]
xtraj = cfill(26,N-1)
constraint_violation = cfill(26,N-1)
total_iter = 0
for i = 1:15
    xtraj, utraj, K, iter = al_ilqr(x0,utraj,xf,Q,Qf,R,N,dt,μ,λ)
    total_iter += iter
    # penalty update for control constraints
    for k = 1:(N-1)
        λ[k] = Π( λ[k] - μ*c_fx(xtraj[k],utraj[k]) )
    end
    # penalty update for goal constraint
    λ[N] = λ[N] - μ*(xtraj[N]-xf)

    # constraint is such that c_fx()<0, so if it's greater than 0 we violate
    for i = 1:length(utraj)
        constraint_violation[i] = c_fx(xtraj[i],utraj[i])
    end
    c_max = max(0,maximum(maximum.(constraint_violation)))

    # include goal constraint as well
    c_max = max(c_max,maximum(abs.(xtraj[N] - xf)))

    outer_loop_solver_logging(i,total_iter,c_max,μ,ϕ)
    # if (c_max!=0 && c_max < params.ϵ_c)
    if (c_max < params.ϵ_c)
        break
    else
        # increase penalty on augmented lagrangian terms
        μ*=ϕ
    end
end


xm = mat_from_vec(xtraj)

X_rbd = [ [q_from_p(xtraj[i][1:3]);xtraj[i][4:end]] for i = 1:length(xtraj)]
um = mat_from_vec(utraj)
mat"
figure
hold on
title('State History')
plot($xm')
hold off
"
mat"
figure
hold on
title('Control History')
plot($um')
hold off
"
return xm, um, X_rbd
end
xm2, um2, X_rbd = runit()


# 3D visualization using MeshCat
ts = [(i-1)*.2 for i = 1:length(X_rbd)]
open(vis)
qs = [X_rbd[i][1:14] for i = 1:length(X_rbd)]
MeshCatMechanisms.animate(vis, ts, qs; realtimerate = 5.)
