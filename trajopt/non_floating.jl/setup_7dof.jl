




urdf = "/Users/kevintracy/devel/SatelliteServicing/7DofArm.urdf"
mechanism = parse_urdf(urdf,gravity = [0.0,0.0,0.0])
state = MechanismState(mechanism)
world = root_frame(mechanism)
body = findbody(mechanism, "link_6")
controlled_tip = Point3D(default_frame(body), 0., 0, 0)

# desired_tip_location = Point3D(world, 4, 1, 1)
desired_tip_location = Point3D(world, 5.356240184435604, 1.3865854372501718, -0.5935291737635336)


# here is how to get the controlled point in the world frame
# transform(state, controlled_tip,world)
# state_initial = MechanismState(mechanism)
# set_configuration!(state_initial,.1*ones(7))

const statecache5 = StateCache(mechanism)

function cost(x::AbstractVector{T}) where T
    """This costs the squared distance between controlled_tip, and
    desired_tip_location

    RBD state:
    _________
        θ1 (1)
        θ2 (1)
    q   θ3 (1)
        θ4 (1)
        θ5 (1)
        θ6 (1)
        θ7 (1)
    _________
        θ̇1 (1)
        θ̇2 (1)
    v   θ̇3 (1)
        θ̇4 (1)
        θ̇5 (1)
        θ̇6 (1)
        θ̇7 (1)
    _________

    """
    state = statecache5[T]
    copyto!(state,x)
    err = ((transform(state, controlled_tip,world) - desired_tip_location).v)
    return dot(err,err)
end

function dynamics(x::AbstractVector{T},u,t) where T
    """Incoming state is the following:
     x =  mrp, position,   angular velocity, velocity
         {      q       } {             v            }
    """
    # now we convert it to a state for RBD
    state = statecache5[T]
    copyto!(state,x)

    # get the dynamics for v (this state is the same for both)
    M = mass_matrix(state)
    if hasnan(M)
        return NaN*x
    else
        v̇ = M\(-dynamics_bias(state) + u)

        # ode's
        return [x[8:14];v̇]
    end
end
