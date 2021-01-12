using RigidBodyDynamics
using StaticArrays
using MeshCatMechanisms
using LinearAlgebra
urdf = "/Users/kevintracy/devel/SatelliteServicing/simple_sat/simple_sat.xml"

mechanism = parse_urdf(urdf,gravity = [0; 0; 0],floating = true)
state = MechanismState(mechanism)

body = findbody(mechanism, "second_arm")

controlled_point = Point3D(default_frame(body), 0., 0, 2)

vis = MechanismVisualizer(mechanism, URDFVisuals(urdf))

# setelement!(vis, controlled_point, 0.2)
#
# world = root_frame(mechanism)
#
# # here is where our controlled_point is currently
# transform(state, controlled_point, world)
#
# # here is where we want it
# desired_tip_location = Point3D(root_frame(mechanism), 0, 0, 5)
#
# function jacobian_transpose_ik!(state::MechanismState,
#                                body::RigidBody,
#                                point::Point3D,
#                                desired::Point3D;
#                                α=0.1,
#                                iterations=1000)
#     mechanism = state.mechanism
#     world = root_frame(mechanism)
#
#     # Compute the joint path from world to our target body
#     p = path(mechanism, root_body(mechanism), body)
#     # Allocate the point jacobian (we'll update this in-place later)
#     Jp = point_jacobian(state, p, transform(state, point, world))
#
#     q = copy(configuration(state))
#
#     for i in 1:iterations
#         # Update the position of the point
#         point_in_world = transform(state, point, world)
#         # Update the point's jacobian
#         point_jacobian!(Jp, state, p, point_in_world)
#         # Compute an update in joint coordinates using the jacobian transpose
#         Δq = Array(Jp)\(transform(state, desired, world) - point_in_world).v
#         # Apply the update
#         q .= configuration(state) .+ α*Δq
#         set_configuration!(state, q)
#
#         if norm(Δq) <1e-6
#             @info "took $i iterations"
#             break
#         end
#     end
#     state
# end
#
# set_configuration!(state,[0.4;-.4;0.4])
# jacobian_transpose_ik!(state, body, controlled_point, desired_tip_location)
#
# set_configuration!(state,[0.3;-.3;-.3])
# set_configuration!(vis, configuration(state))


# dynamics stuff
const statecache2 = StateCache(mechanism)

# function cost(x::AbstractVector{T}) where T
#     """This costs the squared distance between controlled_tip, and
#     desired_tip_location
#
#     RBD state:
#     _________
#         θ1 (1)
#         θ2 (1)
#     q   θ3 (1)
#     _________
#         θ̇1 (1)
#         θ̇2 (1)
#     v   θ̇3 (1)
#     _________
#
#     """
#     state = statecache[T]
#     copyto!(state,x)
#     err = ((transform(state, controlled_tip,world) - desired_tip_location).v)
#     return dot(err,err)
# end

function dynamics(x::AbstractVector{T},u,t) where T
    """Incoming state is the following:
     x =  mrp, position,   angular velocity, velocity
         {      q       } {             v            }

         # solver state
         configuration
         p = x[1:3] <: q
         r = x[4:6] <: q
         θ = x[7:9] <: q

         Velocity
         ω = x[10:12]
         vel = x[13:15]
         θ̇ = x[16:18]


    """

    # solver state
    # configuration stuff
    p = x[1:3]
    r = x[4:6]
    θ = x[7:9]

    # velocity stuff
    ω = x[10:12]
    vel = x[13:15]
    θ̇ = x[16:18]

    # now we convert it to a state for RBD
    state = statecache2[T]
    x_rbd = [q_from_p(p);x[4:end]]
    copyto!(state,x_rbd)

    # get the dynamics for v (this state is the same for both)
    M = Array(mass_matrix(state))
    if hasnan(M)
        return NaN*x
    else
        # dynamics
        @infiltrate
        error()
        v̇ = (M)\(-dynamics_bias(state) + u)

        # kinematics
        q̇ = [pdot_from_w(p,ω);vel;θ̇]

        return [q̇;v̇]
    end
end

# function dynamics(x::AbstractVector{T},u,t) where T
#     """Incoming state is the following:
#      x =  [q;v]
#     """
#     # now we convert it to a state for RBD
#     state = statecache[T]
#     copyto!(state,x)
#
#     # get the dynamics for v (this state is the same for both)
#     M = mass_matrix(state)
#     if hasnan(M)
#         return NaN*x
#     else
#         v̇ = M\(-dynamics_bias(state) + u)
#         # ode's
#         return [x[4:6];v̇]
#     end
# end


controlled_tip = controlled_point
