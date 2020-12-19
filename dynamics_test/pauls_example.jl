using LinearAlgebra
using RigidBodyDynamics
using MeshCat, MeshCatMechanisms
using Attitude
using MATLAB
using SparseArrays




# change to scalar first convention
function qdot2(q1, q2)
    """Quaternion multiplication, hamilton product, scalar first"""

    v1 = @views q1[SVector(2,3,4)]
    s1 = q1[1]
    v2 = @views q2[SVector(2,3,4)]
    s2 = q2[1]

    return SVector{4}([(s1 * s2 - dot(v1, v2));s1 * v2 + s2 * v1 + cross(v1, v2)])
end





# path to URDF, yours will be different
#TODO: you guys have to update this path to be wherever you put the urdf file
urdf = "/Users/kevintracy/devel/SatelliteServicing/dynamics_test/rando.urdf"

# create a robot out of this urdf
rob = parse_urdf(urdf,gravity = [0.0,0.0,0.0],floating = true)

# turn it into a mechanism
state = MechanismState(rob)

# set the state
set_configuration!(state,[1;0;0;0;0;0;0])
set_velocity!(state,[.2;7;.2;0;.0;.0])

# start visualization
mvis = MechanismVisualizer(rob, URDFVisuals(urdf))
open(mvis)

# simulate for 50 seconds
final_time = 10.0

# ts, qs, vs = simulate(state, final_time,simple_control!; Δt = 1e-3,stabilization_gains = nothing)
ts, qs, vs = simulate(state, final_time; Δt = 1e-3,stabilization_gains = nothing)

MeshCatMechanisms.animate(mvis, ts, qs; realtimerate = 0.3)

vsm = mat_from_vec(vs)

mat"
figure
hold on
plot($ts,$vsm(1:3,:)')
hold off
"

result = DynamicsResult(rob)

dynamics!(result,state)

function my_dynamics(state,x,u,t)
    q = x[1:7]
    v = x[8:13]
    set_configuration!(state,q)
    set_velocity!(state,v)

    v̇ = (mass_matrix(state))\(-dynamics_bias(state))
    # v̇ = [ (J\(-cross(v[1:3],J*v[1:3])));
    #      zeros(3)]

    quat = q[1:4]
    ω = v[1:3]
    qdot = .5*qdot2(quat,[0;ω])
    q̇ = [qdot;v[4:6]]
    return [q̇;v̇]
end

function rk4(f, state,x_n, u, t,dt)

    k1 = dt*f(state,x_n,u,t)
    k2 = dt*f(state,x_n+k1/2, u,t+dt/2)
    k3 = dt*f(state,x_n+k2/2, u,t+dt/2)
    k4 = dt*f(state,x_n+k3, u,t+dt)


    return (x_n + (1/6)*(k1+2*k2+2*k3 + k4))

end

dt = 0.001
X = cfill(13,10000)
x0 = [1;0;0;0;zeros(3);.2;7;.2;zeros(3)]
X[1] = copy(x0)
J = Diagonal([1;2;3])
for i = 1:length(X)-1
    X[i+1] = rk4(my_dynamics,state,X[i],zeros(3),0,dt)
end

Xm = mat_from_vec(X)

mat"
figure
hold on
plot($Xm(8:10,:)')
hold off
"

const statecache = StateCache(rob)
function my_dynamics2(x::AbstractVector{T},u,t) where T
    # q = @SVector x[SVector(1,2,3,4,5,6,7)]
    # v = @SVector x[SVector(8,9,10,11,12,13)]
    q = x[1:7]
    v = x[8:13]
    # q = SVector(x[1],x[2],x[3],x[4],x[5],x[6],x[7])
    # v = SVector(x[8],x[9],x[10],x[11],x[12],x[13])
    state = statecache[T]
    set_configuration!(state,q)
    set_velocity!(state,v)

    v̇ = (mass_matrix(state))\(-dynamics_bias(state))
    # v̇ = [ (J\(-cross(v[1:3],J*v[1:3])));
    #      zeros(3)]

    quat = q[1:4]
    ω = v[1:3]
    qdot = .5*qdot2(quat,[0;ω])
    q̇ = [qdot;v[4:6]]
    return SVector{13}([q̇;(mass_matrix(state))\(-dynamics_bias(state))])
end

fdA_fx(x2) =  my_dynamics2(state,x2,0,0)

xsample = X[200]

using ForwardDiff
const FD = ForwardDiff
FD.jacobian(fdA_fx,xsample)



# IK stuff
