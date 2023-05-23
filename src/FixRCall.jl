module FixRCall

using CondaPkg
using Pkg

const restart_on_error = false
needs_restart = false


function find_rcall_path()
    for (uuid, info) in Pkg.dependencies()
        if info.name == "RCall"
            return info.source
        end
    end
    return nothing
end


function fix_rcall(explicit)
    global needs_restart
    if needs_restart
        throw("You need to restart your Julia session")
    end
    CondaPkg.resolve()
    target_rhome = "$(CondaPkg.envdir())/lib/R"
    rcall_path = find_rcall_path()
    if rcall_path == nothing
        if explicit
            println(stderr, "RCall not found in the current project!")
        else
            # Do nothing because the user will get a more familiar error as soon as RCall is imported
        end
        return
    end
    need_update = false
    try
        include(rcall_path * "/deps/deps.jl")
        need_update = realpath(target_rhome) != realpath(Rhome)
    catch err
        if err isa SystemError
            # RCall has not been built yet
            need_update = true
        else
            rethrow()
        end
    end
    rcall_has_been_loaded = false
    for mod in Base.loaded_modules_array()
        if string(mod) == "RCall"
            rcall_has_been_loaded = true
        end
    end
    if explicit
        if need_update
            println(stderr, "RCall will be updated.")
        else
            println(stderr, "RCall's R_HOME was already correctly configured. Leaving as is...")
        end
    elseif need_update
        println(stderr, "Looks like RCall is not pointing to the correct R_HOME. This can happen if e.g. you move your project.")
        println(stderr, "RCall will be updated.")
        if rcall_has_been_loaded
            if restart_on_error
                println(stderr, "Because RCall has already been loaded, this script will attempt to restart itself after RCall is updated.")
            else
                println(stderr, "Because RCall has already been loaded, you will need to restart your script when it is updated.")
            end
        end
    end
    if need_update
        ENV["R_HOME"] = target_rhome
        Pkg.build("RCall")
        if rcall_has_been_loaded && !explicit
            if restart_on_error
                argv = Base.julia_cmd().exec
                opts = Base.JLOptions()
                if opts.project != C_NULL
                    push!(argv, "--project=$(unsafe_string(opts.project))")
                end
                if opts.nthreads != 0
                    push!(argv, "--threads=$(opts.nthreads)")
                end
                @ccall execv(argv[1]::Cstring, argv::Ref{Cstring})::Cint
            else
                throw("You need to restart your Julia session")
                needs_restart = true
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    fix_rcall(true)
else
    fix_rcall(false)
end

end
