using LinearAlgebra
using RigidBodyDynamics
using MeshCat, MeshCatMechanisms
using Attitude
using MATLAB
using SparseArrays
using StaticArrays


# path to URDF, yours will be different
#TODO: you guys have to update this path to be wherever you put the urdf file
urdf = "/Users/kevintracy/devel/SatelliteServicing/dynamics_test/rando.urdf"

# create a robot out of this urdf
rob = parse_urdf(urdf,gravity = [0.0,0.0,0.0],floating = true)

# turn it into a mechanism
state = MechanismState(rob)

# set the state
set_configuration!(state,[1;0;0;0;0;0;0])
set_velocity!(state,[.2;.3;.2;0;.0;.0])

# start visualization
# mvis = MechanismVisualizer(rob, URDFVisuals(urdf))
# open(mvis)

# simulate for 50 seconds
final_time = 8

# ts, qs, vs = simulate(state, final_time,simple_control!; Δt = 1e-3,stabilization_gains = nothing)
ts, qs, vs = simulate(state, final_time; Δt = 1e-3,stabilization_gains = nothing)

# MeshCatMechanisms.animate(mvis, ts, qs; realtimerate = 0.3)

vsm = mat_from_vec(vs)

mat"
figure
hold on
plot($ts,$vsm(1:3,:)')
hold off
"


const statecache = StateCache(rob)
function my_dynamics2(x::AbstractVector{T},u,t) where T
    """Incoming state is the following:
     x =  mrp, position,   angular velocity, velocity
         {      q       } {             v            }
    """

    # this is our state for the ODE
    p = x[1:3]
    r = x[4:6]
    ω = x[7:9]
    vel = x[10:12]

    # now we convert it to a state for RBD
    state = statecache[T]
    copyto!(state,[q_from_p(p);x[4:12]])

    # get the dynamics for v
    v̇ = (mass_matrix(state))\(-dynamics_bias(state))

    # configuration kinematics kinematics
    q̇ = [pdot_from_w(p,ω);vel]

    # ode's
    return [q̇;v̇]
end
function rk42(f,x_n, u, t,dt)

    k1 = dt*f(x_n,u,t)
    k2 = dt*f(x_n+k1/2, u,t+dt/2)
    k3 = dt*f(x_n+k2/2, u,t+dt/2)
    k4 = dt*f(x_n+k3, u,t+dt)

    return (x_n + (1/6)*(k1+2*k2+2*k3 + k4))

end
dt = 0.001
X = cfill(12,length(qs)-2)
x0 = [0;0;0;zeros(3);.2;.3;.2;zeros(3)]
X[1] = copy(x0)
J = Diagonal([1;2;3])
for i = 1:length(X)-1
    X[i+1] = rk42(my_dynamics2,X[i],zeros(3),0,dt)
end

theta_error = zeros(length(X))
for i = 1:length(X)
    q_rbd = qs[i][1:4]
    q_my = q_from_p(X[i][1:3])

    theta_error[i] = norm(phi_from_q(qconj(q_rbd) ⊙ q_my))
end

Xm = mat_from_vec(X)

mat"
figure
hold on
plot($Xm(7:9,:)')
hold off
"

mat"
figure
hold on
plot($theta_error)
hold off
"





# fdA_fx(x2) =  my_dynamics2(state,x2,0,0)
#
# xsample = X[200]
#
# using ForwardDiff
# const FD = ForwardDiff
# FD.jacobian(fdA_fx,xsample)
#
#
#
# # IK stuff
