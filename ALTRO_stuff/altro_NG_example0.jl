using RigidBodyDynamics
using StaticArrays
using Attitude
using LinearAlgebra
using TrajectoryOptimization
using Altro
using RobotDynamics
using MATLAB
const TO = TrajectoryOptimization

struct FloatingSat{T} <: TO.AbstractModel
    mass::T
end
# dynamics stuff
function RobotDynamics.dynamics(model::FloatingSat, x::AbstractVector{T}, u) where T
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

urdf = "/Users/kevintracy/devel/SatelliteServicing/NG_sat_floating/7DofArm.xml"

mechanism = parse_urdf(urdf,gravity = [0; 0; 0],floating = true)
# const statecache = StateCache(mechanism)

Base.size(::FloatingSat) = 26,13

# Model and discretization
model = FloatingSat(1.0)
n,m = size(model)
tf = 30.0  # sec
N = 300    # number of knot points

# Objective
xf = SVector{n}(zeros(n))  # initial state
x0 = SVector{n}([.414;.3;.1;5;2;3;.1*ones(7);zeros(13)])

# Q = Diagonal([10*ones(3);10*ones(3);10*ones(3);.1*ones(3);.1*ones(3);.1*ones(3)])
Q = 100*Diagonal(@SVector ones(n))
Qf = 100*Q
# R = .1*Diagonal([ones(3);ones(3);50*ones(3)])
R = Diagonal(@SVector ones(m))
obj = LQRObjective(Q, R, Qf, xf, N)

# Constraints
cons = ConstraintList(n,m,N)
# add_constraint!(cons, GoalConstraint(xf), N)
add_constraint!(cons, BoundConstraint(n,m, u_min=-50.0, u_max=50.0), 1:N-1)

# Create and solve problem
prob = Problem(model, obj, xf, tf, x0=x0, constraints=cons)

opts = SolverOptions(verbose =2,
                     static_bp = false)
solver = ALTROSolver(prob,opts)
cost(solver)           # initial cost
solve!(solver)         # solve with ALTRO
max_violation(solver)  # max constraint violation
cost(solver)           # final cost
iterations(solver)     # total number of iterations

# Get the state and control trajectories
X = states(solver)
U = controls(solver)

Xm = mat_from_vec(X)

Um = mat_from_vec(U)

mat"
figure
title('X')
hold on
plot($Xm')
hold off
"

mat"
figure
hold on
title('U')
plot($Um')
hold off
"
