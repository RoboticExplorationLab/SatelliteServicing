using ForwardDiff, LinearAlgebra, MATLAB, Infiltrator
using Attitude, StaticArrays
# using FiniteDiff
using DiffResults
# const DR = DiffResults
const FD = ForwardDiff
# const FD2 = FiniteDiff

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
    return .5*(x-xf)'*Q*(x - xf) + .5*u'*R*u
end
function c_fx(x,u)
    # this is for c < 0
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

function ilqr(x0,utraj,xf,Q_lqr,Qf_lqr,R_lqr,N,dt,μ,λ)

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
    J += .5*(xtraj[N]-xf)'*Qf_lqr*(xtraj[N] - xf)

    # allocate K and l
    K = cfill(Nu,Nx,N-1)
    l = cfill(Nu,N-1)

    # allocate the new states and controls
    xnew = cfill(Nx,N)
    unew = cfill(Nu,N-1)

    # main loop
    for iter = 1:50

        # Cost to go hessian and gradient at final time step
        S = copy(Qf_lqr)
        s = Qf_lqr*(xtraj[N] - xf)

        # backwards pass
        for k = (N-1):-1:1

            # ∂²J/∂x²
            Q = Q_lqr

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

            # solve the linear system for feedforward (l) and feedback gain (K)
            F = factorize(R + Bk'*S*Bk)
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
            Jnew += .5*(xnew[N]-xf)'*Qf_lqr*(xnew[N] - xf)

            # if the new cost is lower, we keep the new trajectory
            if Jnew<J
                break
            else# this also pulls the linesearch back if hasnan(xnew)
                α /= 2
            end

            # this will happen if linearization is bad for some reason, or
            # if the trajectory has converged and can't do any better
            if inner_i == 20
                @warn ("Line Search Failed")
                @show J
                @show Jnew
                Jnew = copy(J)
                break
            end

        end

        # update trajectory and control history
        xtraj = copy(xnew)
        utraj = copy(unew)

        # termination criteria
        DJ = abs(J - Jnew)
        if DJ<params.dJ_tol
            break
        end

        # ----------------------------output stuff-----------------------------
        if rem((iter-1),4)==0
            println("iter       alpha      maxL    Cost      DJ")
        end
        DJ = round(DJ,sigdigits = 3)
        maxL = round(maximum(maximum.(l)),sigdigits = 3)
        J = Jnew
        J_display = round(J,sigdigits = 3)
        alpha_display = round(α,sigdigits = 3)
        println("$iter          $alpha_display      $maxL    $J_display    $DJ")


    end

    return xtraj, utraj, K
end


function runit()

u_min = -3*(@SVector ones(9))
u_max = 3*(@SVector ones(9))
Q = Diagonal([100*ones(3);10*ones(3);10*ones(3);.1*ones(3);.1*ones(3);.1*ones(3)])
Qf = 100*Q
R = .1*Diagonal([ones(3);ones(3);50*ones(3)])
x0 = zeros(18)

# solver state
# x = [mrp; position; joint angles; ω; vel; joint speeds]

# RBD state (purely inside the dynamics function)
# x = [quaternion; position; joint angles; ω; vel; joint speeds]

xf = [.414;.3;.1;5;2;3;[pi;deg2rad(45);-deg2rad(90)];zeros(9)]
global params = (dJ_tol = 1e-4, u_min = u_min, u_max = u_max,ϵ_c = 1e-4)
dt = 0.2
N = 75


μ = 1
λ = cfill(18,N-1)
utraj = [0*randn(9) for i = 1:N-1]
xtraj = cfill(18,N-1)
constraint_violation = cfill(18,N-1)
for i = 1:5
    xtraj, utraj, K = ilqr(x0,utraj,xf,Q,Qf,R,N,dt,μ,λ)

    # penalty update
    for k = 1:length(λ)
        λ[k] = Π( λ[k] - μ*c_fx(xtraj[k],utraj[k]) )
    end
    @show "update done"


    for i = 1:length(utraj)
        constraint_violation[i] = c_fx(xtraj[i],utraj[i])
    end
    max_con_vi = maximum(maximum.(constraint_violation))

    @show max_con_vi
    # umm = mat_from_vec(utraj)
    # umin = minimum(umm)
    # umax = maximum(umm)
    #
    # top_violate = max(0,umax - params.u_max[1])
    # bot_violate = max(0,params.u_min[1] - umin)
    #
    # @show max(top_violate,bot_violate)
    # for ii = 1:(length(xtraj)-1)
    #     constraint_violation[:,i] = max.(0,c_fx(xtraj[ii],utraj[ii]))
    # end
    # max_c = maximum(vec(constraint_violation))
    # @show max_c
    μ*=5
end


xm = mat_from_vec(xtraj)

X_rbd = [ [q_from_p(xtraj[i][1:3]);xtraj[i][4:end]] for i = 1:length(xtraj)]
um = mat_from_vec(utraj)
mat"
figure
hold on
plot($xm')
hold off
"
mat"
figure
hold on
plot($um')
hold off
"
return xm, um, X_rbd
end
xm2, um2, X_rbd = runit()


ts = [(i-1)*.2 for i = 1:length(X_rbd)]
open(vis)
#
qs = [X_rbd[i][1:10] for i = 1:length(X_rbd)]
MeshCatMechanisms.animate(vis, ts, qs; realtimerate = 2.)
