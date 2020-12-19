using RigidBodyDynamics
using StaticArrays
using MeshCatMechanisms, Blink

using Random
Random.seed!(42);

srcdir = dirname(pathof(RigidBodyDynamics))
urdf = joinpath(srcdir, "..", "test", "urdf", "Acrobot.urdf")
mechanism = parse_urdf(urdf)
state = MechanismState(mechanism)

body = findbody(mechanism, "lower_link")
point = Point3D(default_frame(body), 0., 0, -2)

# Create the visualizer
vis = MechanismVisualizer(mechanism, URDFVisuals(urdf))

# Render our target point attached to the robot as a sphere with radius 0.07
setelement!(vis, point, 0.07)

open(mvis)
mvis = MechanismVisualizer(mechanism, URDFVisuals(urdf));

function jacobian_transpose_ik!(state::MechanismState,
                               body::RigidBody,
                               point::Point3D,
                               desired::Point3D;
                               α=0.1,
                               iterations=100)
    mechanism = state.mechanism
    world = root_frame(mechanism)

    # Compute the joint path from world to our target body
    p = path(mechanism, root_body(mechanism), body)
    # Allocate the point jacobian (we'll update this in-place later)
    Jp = point_jacobian(state, p, transform(state, point, world))

    q = copy(configuration(state))

    for i in 1:iterations
        # Update the position of the point
        point_in_world = transform(state, point, world)
        # Update the point's jacobian
        point_jacobian!(Jp, state, p, point_in_world)
        # Compute an update in joint coordinates using the jacobian transpose
        Δq = α * Array(Jp)' * (transform(state, desired, world) - point_in_world).v
        # Apply the update
        q .= configuration(state) .+ Δq
        set_configuration!(state, q)
    end
    state
end

rand!(state)
set_configuration!(vis, configuration(state))

desired_tip_location = Point3D(root_frame(mechanism), 0.5, 0, 2)


controlled_tip = point

desired_tip_location

world = root_frame(mechanism)

# here is how to get the controlled point in the world frame
transform(state, controlled_tip,world)

# here is how we get the error
err = (transform(state, controlled_tip,world) - desired_tip_location).v

q1 = [1;2.0]
v1 = [3;4.0]


const statecache2 = StateCache(mechanism)

function cost(x::AbstractVector{T}) where T
    state = statecache2[T]
    copyto!(state,x)
    err = ((transform(state, controlled_tip,world) - desired_tip_location).v)
    return transpose(err)*err
end

# forward diff test
FD.hessian(cost,randn(4))
