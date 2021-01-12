using ForwardDiff, LinearAlgebra, MATLAB, Infiltrator
using Attitude, StaticArrays, FiniteDiff
const FD = ForwardDiff
const FD2 = FiniteDiff
# function dynamics(x,u,t)
#
#     p = SVector(x[1],x[2],x[3])
#     ω = SVector(x[4],x[5],x[6])
#
#     J = @views params.J
#     invJ = @views params.invJ
#
#     return SVector{6}([pdot_from_w(p,ω);
#            invJ*(u - ω × (J*ω))])
# end
function rk4(f, x_n, u, t,dt)

    # x_n = SVector{6}(x_n)

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
    # return (.5*cost(x) + .5*u'*R*u)
end
function c_fx(x,u)
    # this is for c < 0
    # u <= u_max
    # u - u_max <= 0
    # u >= u_min
    #-u <= u_min
    #-u + u_min
    return [u - params.u_max;
           -u + params.u_min]
end

function Π(z)
    out = zeros(eltype(z),length(z))
    for i = 1:length(out)
        out[i] = min(0,z[i])
    end

    return out
end


function Lag(Q,R,x,u,xf,λ,μ)
    # return (LQR_cost(Q,R,x,u,xf) +
    #         (1/(2*μ))*(  norm(Π(λ - μ*c_fx(x,u)))^2 - dot(λ,λ)))
    return (LQR_cost(Q,R,x,u,xf))
end
# function Lag2(Q,R,x,u,xf,λ,μ)
#     @infiltrate
#     error()
#     return (LQR_cost(Q,R,x,u,xf) +
#             (1/(2*μ))*(  norm(Π(λ - μ*c_fx(x,u)))^2 - dot(λ,λ)))
# end
function ilqr(x0,utraj,xf,Q_lqr,Qf_lqr,R_lqr,N,dt,μ,λ)

    # state and control dimensions
    # Nx, Nu = size(Bd)
    Nx = length(x0)
    Nu = size(R_lqr,1)

    # initial trajectory is initial conditions the whole time
    xtraj = cfill(Nx,N)
    xtraj[1] = copy(x0)
    # utraj = cfill(Nu,N-1)

    J = 0.0
    for k = 1:N-1
        # @infiltrate
        # error()
        xtraj[k+1] = discrete_dynamics(xtraj[k],utraj[k],(k-1)*dt,dt)
        # J += LQR_cost(Q,R,xtraj[k],utraj[k],xf)
        J += Lag(Q_lqr,R_lqr,xtraj[k],utraj[k],xf,λ[k],μ)
    end
    J += .5*(xtraj[N]-xf)'*Qf_lqr*(xtraj[N] - xf)
    # J += .5*cost(xtraj[N])

    @info "rollout done"
    @show J
    # @show J
    # allocate K and l
    K = cfill(Nu,Nx,N-1)
    l = cfill(Nu,N-1)

    # allocate the new states and controls
    xnew = cfill(Nx,N)
    unew = cfill(Nu,N-1)

    # main loop
    for iter = 1:50

        # cost to go matrices at the end of the trajectory
        # S = Qf_lqr
        # s = Qf_lqr*(xtraj[N]-xf)
        # _L_fx(xl) = Lag(Q_lqr,R_lqr,xl,utraj[1],xf,λ[1],μ)

        # @infiltrate
        # error()
        L_fx(xl) = Lag(Q_lqr,R_lqr,xl,utraj[1],xf,λ[1],μ)
        S = FD2.finite_difference_hessian(L_fx,xtraj[N])
        s = FD2.finite_difference_gradient(L_fx,xtraj[N])
        # S = .5*FD2.finite_difference_hessian(cost,xtraj[N])
        # s = .5*FD2.finite_difference_gradient(cost,xtraj[N])

        # @info "first gradients"
        # backwards pass
        for k = (N-1):-1:1

            # calculate cost gradients for this time step
            # q = Q*(xtraj[k]-xf)
            # r = R*utraj[k]
            __L_fx(xl) = Lag(Q_lqr,R_lqr,xl,utraj[k],xf,λ[k],μ)

            # @infiltrate
            # error()
            q = FD2.finite_difference_gradient(__L_fx,xtraj[k])
            # @info "gradient"
            Q = FD2.finite_difference_hessian(__L_fx,xtraj[k])
            # @info "hessian"
            # error()
            # Q = FD.hessian(cost,xtraj[k])
            # q = FD.gradient(cost,xtraj[k])
            #
            # # Lag2(Q,R,xtraj[k],utraj[k],xf,λ[k],μ)
            _L_fx2(ul) = Lag(Q_lqr,R_lqr,xtraj[k],ul,xf,λ[k],μ)
            R = FD2.finite_difference_hessian(_L_fx2,utraj[k])
            # @info "R Hessian"
            r = FD2.finite_difference_gradient(_L_fx2,utraj[k])
            # R = copy(R_lqr)
            # r = R_lqr*utraj[k]

            # @info "got cost gradients"
            # error()


            # @infiltrate
            # error()
            # discrete dynamics jacobians
            _A_fx(x) = discrete_dynamics(x,utraj[k],(k-1)*dt,dt)
            Ak = FD2.finite_difference_jacobian(_A_fx,xtraj[k])

            # @info "A matrix"
            __B_fx(u_loc) =  discrete_dynamics(xtraj[k],u_loc,(k-1)*dt,dt)
            # @infiltrate
            # error()
            Bk = FD2.finite_difference_jacobian(__B_fx,utraj[k])

            # @info "B matrix"
            # error()
            # linear solve
            l[k] = (R + Bk'*S*Bk)\(r + Bk'*s)
            K[k] = (R + Bk'*S*Bk)\(Bk'*S*Ak)

            # update
            Snew = Q + K[k]'*R*K[k] + (Ak-Bk*K[k])'*S*(Ak-Bk*K[k])
            snew = q - K[k]'*r + K[k]'*R*l[k] + (Ak-Bk*K[k])'*(s - S*Bk*l[k])

            # update S's
            S = copy(Snew)
            s = copy(snew)
            # @show k

        end

        # initial conditions
        xnew[1] = copy(x0)

        # learning rate
        alpha = 1.0

        # forward pass
        Jnew = 0.0
        for inner_i = 1:10
            # @show inner_i
            Jnew = 0.0
            # rollout the dynamics
            for k = 1:N-1
                unew[k] = utraj[k] - alpha*l[k] - K[k]*(xnew[k]-xtraj[k])
                xnew[k+1] = discrete_dynamics(xnew[k],unew[k],(k-1)*dt,dt)
                # Jnew += LQR_cost(Q,R,xnew[k],unew[k],xf)
                Jnew += Lag(Q_lqr,R_lqr,xnew[k],unew[k],xf,λ[k],μ)
            end
            Jnew += .5*(xnew[N]-xf)'*Q_lqr*(xnew[N] - xf)
            # Jnew += .5*cost(xnew[N])

            # if the new cost is lower, we keep it
            # @show Jnew
            # @show J
            if Jnew<J
                break
            else# this also pulls the linesearch back if hasnan(xnew)
                alpha = (1/2)*alpha
            end
            if inner_i == 10
                @warn ("Line Search Failed")
                Jnew = copy(J)
                break
            end

        end

        # update trajectory and control history
        xtraj = copy(xnew)
        utraj = copy(unew)

        # termination criteria
        if abs(J - Jnew)<params.dJ_tol
            break
        end


        # ----------------------------output stuff-----------------------------
        if rem((iter-1),4)==0
            println("iter       alpha      maxL    Cost")
        end
        maxL = round(maximum(vec(mat_from_vec(l))),digits = 3)
        J = Jnew
        J_display = round(J,digits = 3)
        alpha_display = round(alpha,digits = 3)
        println("$iter          $alpha_display      $maxL    $J_display")


    end

    return xtraj, utraj, K
end


function runit()
# phi_0 = deg2rad(140)*normalize(randn(3))
# x0 = SVector{6}([p_from_phi(phi_0);0;0;0])
# xf = @SVector zeros(6)
# Q = Diagonal(@SVector ones(6))
# Qf = 10*Q
# R = Diagonal(@SVector ones(3))
u_min = -0.05*(@SVector ones(3))
u_max = 0.05*(@SVector ones(3))
Q = Diagonal(SVector{14}([ones(7);0.000001*ones(7)]))
Qf = 10*Q
xf = randn(14)
R = .001*Diagonal(@SVector ones(7))
x0 = zeros(14)
x0 = 0.1*randn(14)
xf = zeros(14)
global params = (dJ_tol = 1e-6, u_min = u_min, u_max = u_max,ϵ_c = 1e-4)
dt = 0.1
N = 50


μ = 1e-2
λ = cfill(6,N-1)
# utraj = cfill(7,N-1)
utraj = [0.00*randn(7) for i = 1:N-1]
xtraj = cfill(14,N-1)
constraint_violation = zeros(14,N-1)
for i = 1:1
    xtraj, utraj, K = ilqr(x0,utraj,xf,Q,Qf,R,N,dt,μ,λ)
    # # @show "done with 1"
    # # penalty update
    # for k = 1:length(λ)
    #     # @show k
    #
    #     # z = λ[k] - μ*c_fx(xtraj[k],utraj[k])
    #     # @infiltrate
    #     λ[k] = Π( λ[k] - μ*c_fx(xtraj[k],utraj[k]) )
    # end
    # # @show "update done"
    # # for ii = 1:(length(xtraj)-1)
    # #     constraint_violation[:,i] = max.(0,c_fx(xtraj[ii],utraj[ii]))
    # # end
    # # max_c = maximum(vec(constraint_violation))
    # # @show max_c
    # μ*=5
end


xm = mat_from_vec(xtraj)
um = mat_from_vec(utraj)
mat"
figure
hold on
title('X Configuration')
plot($xm(1:7,:)')
hold off
"
mat"
figure
hold on
title('X Velocity')
plot($xm(8:14,:)')
hold off
"
mat"
figure
hold on
title('U')
plot($um')
hold off
"
end
runit()
