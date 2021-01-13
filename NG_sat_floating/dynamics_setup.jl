using RigidBodyDynamics
using StaticArrays
using MeshCatMechanisms
using LinearAlgebra

urdf = joinpath(dirname(@__FILE__),"7DofArm.xml")

mechanism = parse_urdf(urdf,gravity = [0; 0; 0],floating = true)

vis = MechanismVisualizer(mechanism, URDFVisuals(urdf))

const statecache = StateCache(mechanism)

function dynamics(x::AbstractVector{T},u,t) where T
    """Incoming state is the following:

         # solver state
         configuration
         p = x[1:3] <: q
         r = x[4:6] <: q
         θ = x[7:13] <: q

         Velocity
         ω = x[14:16]
         vel = x[17:19]
         θ̇ = x[20:26]


    """

    # solver state
    # configuration stuff
    p = x[1:3]
    r = x[4:6]
    θ = x[7:13]

    # velocity stuff
    ω = x[14:16]
    vel = x[17:19]
    θ̇ = x[20:26]

    # now we convert it to a state for RBD
    """
    RigidBodyDynamics.jl state
    x = [q;v]
    q = [quaternion;position;join angles] {n+1}
    v = [ω; vel; joint speeds]            {n}
    """
    state = statecache[T]
    x_rbd = [q_from_p(p);x[4:end]]
    copyto!(state,x_rbd)

    # get the dynamics for v (this state is the same for both)
    M = Array(mass_matrix(state))
    if hasnan(M)
        # if there is NaN anywhere, return NaN's instead of erroring
        return NaN*x
    else
        # dynamics
        v̇ = (M)\(-dynamics_bias(state) + u)

        # kinematics
        q̇ = [pdot_from_w(p,ω);vel;θ̇]

        return [q̇;v̇]
    end
end
