struct ModelDiffBuffer{p, Dg, Dh, Pg, Ph, tg, Tg, Th, c}
    dr::DiffBase.DiffResult{2,Float64,Tuple{Array{Float64,1},Array{Float64,2}}}
    gc::ForwardDiff.GradientConfig{Tg,Float64,c,Array{Dg,1}}
    hc::ForwardDiff.HessianConfig{Th,Float64,c,Array{Dh,1},tg,Tuple{Array{Dg,1},Array{Dg,1}}}
    mp_g::ModelParam{p, Dg, Vector{Dg}, Pg}
    mp_h::ModelParam{p, Dh, Vector{Dh}, Ph}
	calls::Vector{Int}
    method::Optim.NewtonTrustRegion{Float64}
    state::Optim.NewtonTrustRegionState{Float64,1,Array{Float64,1}}
    options::Optim.Options{Void}
end
function ModelDiffBuffer(Θ::Tuple, ::Type{Val{p}}, options = Optim.Options(), ::Type{T} = Float64) where {p, T}

	initial_x = zeros(T, p)
	chunk = chunk_size(Val{p})

	dr = DiffBase.DiffResult(zero(T), (Array{T}(p), Array{T}(p,p)) )
	gc = ForwardDiff.GradientConfig(nothing, initial_x, chunk)
	hc = ForwardDiff.HessianConfig(nothing, dr, initial_x, chunk)

	mp_g = construct(Array{eltype(gc.duals)}(p), Θ, Val{p})
	mp_h = construct(Array{eltype(hc.gradient_config.duals)}(p), Θ, Val{p})

	calls = zeros(Int, 2)

    method = NewtonTrustRegion()

    state = NewtonTrustRegionState(initial_x, # Maintain current state in state.x
              Array{T}(p), # Maintain previous state in state.x_previous
              similar(dr.derivs[1]), # Store previous gradient in state.g_previous
              T(NaN), # Store previous f in state.f_x_previous
              Array{T}(p), # Maintain current search direction in state.s
              false,
              true,
              true,
              T(method.initial_delta),
              NaN,
              method.eta, # eta
              zero(T))

    ModelDiffBuffer(dr, gc, hc, mp_g, mp_h, calls, method, state, options)
end
struct ModelDiff{p, B <: ModelDiffBuffer{p}, Fg <: Function, Fh <: Function}
	mdb::B
	ld_g::Fg
	ld_h::Fh
end
function ModelDiff(f::Function, d::B, data) where {p, B <: ModelDiffBuffer{p}}
    #Need to make sure the closures properly catch types.
    let data = data, d = d
        function ld_g(x)
            copy!(d.mp_g.v, x)
		    update!(d.mp_g)
            - log_jacobian(d.mp_g) - f(data, d.mp_g.Θ...)
        end
    	function ld_h(x)
            copy!(d.mp_h.v, x)
    		update!(d.mp_h)
            - log_jacobian(d.mp_h) - f(data, d.mp_h.Θ...)
        end
    end
    fill!(d.calls, 0)
    ModelDiff{p, B, typeof(ld_g), typeof(ld_h)}(d, ld_g, ld_h)
end



function update_g!(d::ModelDiff, x)
	d.mdb.calls[1] += 1
	ForwardDiff.gradient!(d.mdb.dr, d.ld_g, x, d.mdb.gc)
end
function update_h!(d::ModelDiff, x)
	d.mdb.calls .+= 1
	ForwardDiff.hessian!(d.mdb.dr, d.ld_h, x, d.mdb.hc)
end
value(d::ModelDiff) = d.mdb.dr.value
gradient(d::ModelDiff) = d.mdb.dr.derivs[1]
hessian(d::ModelDiff) = d.mdb.dr.derivs[2]


f_calls(d::ModelDiff) = d.mdb.calls[1]
g_calls(d::ModelDiff) = d.mdb.calls[1]
h_calls(d::ModelDiff) = d.mdb.calls[2]
#A type stable adaptation of the chunk-size determining heuristic from ForwardDiff.
@Base.pure function chunk_size(::Type{Val{p}}) where p
	if p <= 10
        return ForwardDiff.Chunk{p}()
    else
        return ForwardDiff.Chunk{round(Int, p / round(Int, p / 10, RoundUp), RoundUp)}()
    end

end

function optimize!(d::ModelDiff{n}, options::Options = d.mdb.options, state::Optim.NewtonTrustRegionState{T,1,Array{T,1}} = d.mdb.state) where {n,T}

    set_state!(d, state)

    t0 = time() # Initial time stamp used to control early stopping by options.time_limit

 #   tr = OptimizationTrace{typeof(method)}()
    tracing = options.store_trace || options.show_trace || options.extended_trace || options.callback != nothing
    stopped, stopped_by_callback, stopped_by_time_limit = false, false, false
    f_limit_reached, g_limit_reached, h_limit_reached = false, false, false
    x_converged, f_converged, f_increased = false, false, false
    g_converged = vecnorm(gradient(d), Inf) < options.g_tol

    converged = g_converged
    iteration = 0

 #   options.show_trace && print_header(method)
 #   trace!(tr, d, state, iteration, method, options)

    while !converged && !stopped && iteration < options.iterations
        iteration += 1

        update_state!(d, state) && break # it returns true if it's forced by something in update! to stop (eg dx_dg == 0.0 in BFGS)
        update_h!(d, state.x) #We do want the Hessian for the next step and at the minimum post-convergence.
        x_converged, f_converged,
        g_converged, converged, f_increased = assess_convergence(state, d, options)

 #       if tracing
 #           # update trace; callbacks can stop routine early by returning true
 #           stopped_by_callback = trace!(tr, d, state, iteration, method, options)
 #       end

        stopped_by_time_limit = time()-t0 > options.time_limit ? true : false
        f_limit_reached = options.f_calls_limit > 0 && f_calls(d) >= options.f_calls_limit ? true : false
        g_limit_reached = options.g_calls_limit > 0 && g_calls(d) >= options.g_calls_limit ? true : false
        h_limit_reached = options.h_calls_limit > 0 && h_calls(d) >= options.h_calls_limit ? true : false

        if (f_increased && !options.allow_f_increases) || #stopped_by_callback ||
            stopped_by_time_limit || f_limit_reached || g_limit_reached || h_limit_reached
            throw("Limit reached.\nTime: $stopped_by_time_limit\nF_limit: $f_limit_reached\nG_limit: $g_limit_reached\nH_limit: $h_limit_reached.")
        end
    end # while

end


function set_state!(d::ModelDiff, state::Optim.NewtonTrustRegionState{T,1,Array{T,1}}) where T
    state.delta = one(T)
    state.rho = zero(T)
    update_h!(d, state.x)
end


function update_state!(d::ModelDiff, state::NewtonTrustRegionState{T}) where T
    # Find the next step direction.
    m, state.interior, state.lambda, state.hard_case, state.reached_subproblem_solution =
        solve_tr_subproblem!(gradient(d), hessian(d), state.delta, state.s)

    # Maintain a record of previous position
    copy!(state.x_previous, state.x)

    # Update current position
    state.x .+= state.s

    # Update the function value and gradient
    copy!(state.g_previous, gradient(d))
    state.f_x_previous = value(d)
    update_g!(d, state.x)


    # Update the trust region size based on the discrepancy between
    # the predicted and actual function values.  (Algorithm 4.1 in N&W)
    f_x_diff = state.f_x_previous - value(d)
    if abs(m) <= eps(T)
        # This should only happen when the step is very small, in which case
        # we should accept the step and assess_convergence().
        state.rho = 1.0
    elseif m > 0
        # This can happen if the trust region radius is too large and the
        # Hessian is not positive definite.  We should shrink the trust
        # region.
        state.rho = d.mdb.method.rho_lower - 1.0
    else
        state.rho = f_x_diff / (0 - m)
    end

    if state.rho < d.mdb.method.rho_lower
        state.delta *= 0.25
    elseif (state.rho > d.mdb.method.rho_upper) && (!state.interior)
        state.delta = min(2 * state.delta, d.mdb.method.delta_hat)
    else
        # else leave delta unchanged.
    end

    if state.rho <= state.eta
        # The improvement is too small and we won't take it.

        # If you reject an interior solution, make sure that the next
        # delta is smaller than the current step.  Otherwise you waste
        # steps reducing delta by constant factors while each solution
        # will be the same.
        x_diff = state.x - state.x_previous
        delta = 0.25 * sqrt(vecdot(x_diff, x_diff))

        d.f_x = state.f_x_previous
        copy!(state.x, state.x_previous)
        copy!(gradient(d), state.g_previous)
    end

    false
end
