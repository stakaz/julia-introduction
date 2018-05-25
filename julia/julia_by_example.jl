using BenchmarkTools

########################################
### Numbers and Strings
########################################
1 # integer
2.0 # float
3//2 # rational
b = big"123456789012345678901234567890" + 1
typeof(b)
s = "String"

s = 'c' #char

3 ≠ 4 # or !=
3 ≥ 4 # or >=
3//3 == 1 == 1.0

### Generators
typeof(1:10)
typeof(linspace(1,2,10))

# (i < 4 for i in [1, 2, 3]) == true
########################################
## #Arrays as Matricies
########################################
## , 			- separator
## space	- horizontal concatination
## ;			- vertical concatination
a = [1 2] # 1x2 2-dimensional Array
b = [5 6]' # 2x1 2-dimensional Array (1x2 Array transposed)
c = [1, 2] # 1-dimensioal Array but [1; 2] is the same, not 2x1 Array
A = [1 2; 3 4] # 2x2 2-dimensional Array

A' # conjugate transposed A
A.' # transposed A
A^-1 # inversed A or inv(A)
A * b # matrix multiplication = column vector
typeof(A * b)
A * c # also works = 1-dim Array
typeof(A * c)
x = A \ b # solution to Ax = b
A * x

### map and filter
B = map(x -> x^2, A)
C = filter(x -> x ≤ 2, A)

### comprehension
B = A.^2
C = A[A .≤ 2]

########################################
## Functions
########################################
## short form on one line
f(x) = 2x

## long form with doc string
"""
	f(x)

This is very handy and useful to write doc strings.
# Example
```
f(2) = 4
```
"""
function f(x)
	2x # with optional return keyword
end

## annonymous functions / lambdas
x -> 2x
## or long form
function (x)
	2x
end

########################################
### C and Python calls
########################################
t = ccall((:clock, "libc"), Int32, ())


using PyCall
@pyimport math
math.sin(math.pi / 4) - sin(pi / 4)

########################################
## Syntactic Sugar
########################################
f(x) = 2x
g(x) = x^2
h(x) = x + 1

(f ∘ g)(3) # f(g(x)) = 2 * (3^2) = 18
(f ∘ g).([3, 4, 5])

((x -> √x) ∘ g)(2)

(sort ∘ unique)([7, 5, 1, 4, 5, 1])

### function addition and multiplication

⊕(f::Function, g::Function) = (x...) -> f(x...) + g(x...)
(f::Function)⊗(g::Function) = (x...) -> f(x...) * g(x...)

(f ⊕ g)(3)
(f ⊗ g)(3)
(h ⊕ f ⊗ g)(3) ## precedence works automatically ;)
(h ⊕ f ⊕ g)(3)


[i^2 for i ∈ [1, 2, 3] if i ≥ 2] ## Array construction with for

(i^2 for i ∈ [1, 2, 3] if i ≥ 2) ## Iterator construction with for

Dict(1 => 2, 3 => 4)
Dict("$i^2" => i^2 for i ∈ [1, 2, 3]) ## Dict construction with for

[1, 2, 3] ∋ 7 # check if in Array, alternative to 7 ∈ S or 7 in S
7 ∉ [1, 2, 3] # alternative to !(7 in S)

## abbreviate functions with greek letters
σ² = var
μ = mean

σ²(x) # variace
μ(x) # mean
μ(A) # mean of all elems
μ(A, 1) # mean along dimension 1
μ(A, 2) # mean along dimension 2
## dimensions are column based !


S = [1, 2, 3, 3, 1, 3]
T = [4, 5, 5, 3]
# union
S ∪ T
### BUT !!!
T ∪ S
# Construct the union of two or more sets.
# Maintains order with arrays.

# intersect
S ∩ T
### BUT !!!
T ∩ S
# Construct the intersection of two or more sets.
# Maintains order and multiplicity of the first
# argument for arrays and ranges.
### Sets
S = Set([1, 2, 3, 3, 1, 3])
T = Set([4, 5, 5, 3])
V = Set([1, 2, 3, 4, 5, 6, 7, 8, 9])

S ∪ T
T ∪ S
S ∩ T
T ∩ S

∅ = Set()

A::Set ⊂ B::Set = all(B ∋ i for i ∈ A)
A::Set ⊃ B::Set = B ⊂ A

∅ ⊂ T
S ⊂ T
S ⊂ V
V ⊃ T
########################################
## Vectorized Operations
########################################

function vectorized(x)
	z = similar(x)
	z .= sin.(x) .+ 2.0 .* cos.(x).^2 ## or @. z = sin(x) + 2.0 * cos(x)^2
	return z
end

function not_fully_vectorized(x)
	z = similar(x)
	z .= sin.(x) + 2.0 * cos.(x).^2
	return z
end

function looped(x)
	z = similar(x)
	for i in eachindex(x)
		z[i] = sin(x[i]) + 2.0 * cos(x[i])^2
	end
	return z
end

N = 500000
R = rand(N);

@btime vectorized(R)
@btime looped(R)
@btime not_fully_vectorized(R)

########################################
### Parallel Processing
########################################
# pi example from https://www.juliabloggers.com/monthofjulia-day-12-parallel-processing/

addprocs(4 - length(procs())) ## add not more than 4 proccesses

function single_findpi(N)
	inside = 0
	for i in 1:N
		x, y = rand(2)
		inside += x^2 + y^2 ≤ 1 ? 1 : 0
	end
	4 * inside / N
end

function parallel_findpi(N)
	inside = @parallel (+) for i in 1:N
		x, y = rand(2)
		x^2 + y^2 ≤ 1 ? 1 : 0
	end
	4 * inside / N
end

@show N1 = 1_000
@btime single_findpi(N1)
@btime parallel_findpi(N1)

@show N2 = 100_000
@btime single_findpi(N2)
@btime parallel_findpi(N2)

@show N3 = 1_000_000
@btime single_findpi(N3)
@btime parallel_findpi(N3)



########################################
### Recursion
########################################

function factorial_rec(x)
	if x > 0
		x * factorial_rec(x - 1)
	else
		1
	end
end

function factorial_loop(x)
	r = 1
	while x > 0
		r *= x
		x = x - 1
	end
	r
end

@btime factorial_rec(big"400")
@btime factorial_loop(big"400")

########################################
### Type Stability
########################################

### concrete types vs Any (like pure Python)

function randlookup(a, N)
	for i in 1:N
		a[1] += 1
	end
end

R = rand(1:10, 10);
any_array = Array{Any}(R);
int_array = Array(R);
N = 100000

@btime randlookup(any_array, N)
@btime randlookup(int_array, N)

unsafe_return_type(x) = x > 0 ? x : 0
safe_return_type(x) = x > 0 ? x : zero(x)

print_with_color(:red, "##########")
@code_llvm     unsafe_return_type(2)
print_with_color(:red, "##########")
@code_llvm     safe_return_type(2)

print_with_color(:red, "##########")
@code_warntype unsafe_return_type(2)
print_with_color(:red, "##########")
@code_warntype safe_return_type(2)

print_with_color(:red, "##########")
@code_llvm     unsafe_return_type(2.0)
print_with_color(:red, "##########")
@code_llvm     safe_return_type(2.0)

print_with_color(:red, "##########")
@code_warntype unsafe_return_type(2.0)
print_with_color(:red, "##########")
@code_warntype safe_return_type(2.0)

function unsafe_local_type(x)
	r = 1
	if x > 0
		r = x
	end
	return r
end

function safe_local_type(x)
	r = convert(typeof(x), 1)
	if x > 0
		r = x
	end
	return r
end

@code_llvm     unsafe_local_type(2)
@code_llvm     safe_local_type(2)
@code_warntype unsafe_local_type(2)
@code_warntype safe_local_type(2)

@code_llvm     unsafe_local_type(2.0)
@code_llvm     safe_local_type(2.0)
@code_warntype unsafe_local_type(2.0)
@code_warntype safe_local_type(2.0)

@btime unsafe_local_type(0.1)
@btime safe_local_type(0.1)


########################################
### data frames and plotting
########################################

using DataFrames

D = DataFrame(
	label = ["apple", "apple", "orange", "kiwi", "kiwi"],
	radius = [3.5, 3.7, 4.5, 2.3, 3.1],
	weight = [1.2, 1.3, 2.0, 0.9, 1.0],
	taste = [3, 4, 2, 5, 4],
)

fruit_colors = Dict("apple" => :red, "orange" => :orange, "kiwi" => :green)

D[:label] ## access columns

D[D[:label] .== "kiwi", :] ## filter by condition

sort!(D, cols=[:radius, :weight]) ## sorting

D[[:label, :weight]] ## select multiple columns

using PyPlot

plt[:style][:use]("/home/gluon/.config/matplotlib/stas_poster.mplstyle")

for i in unique(D[:label])
	tmp = D[D[:label] .== i, :]
	scatter(tmp[:weight], tmp[:radius], color=fruit_colors[i], label = i)
	legend()
	xlabel("weight")
	ylabel("radius")
end

x = [1 2 3 ; 4 5 6]
y = [3 4 5 ; 7 8 9]
plot(x, y)


########################################
### Slicing is a View by Default
########################################


expansive_slice_operation(x) = maximum(x[:, 5:2:9, 3:5])
expansive_slice_operation_view(x) = maximum(@view x[:, 5:2:9, 3:5])

for N in [200, 100, 10]
	@show N
	R = rand(N,N,N)

	@btime copied = expansive_slice_operation(R)
	@btime viewed = expansive_slice_operation_view(R)
end

expansive_slice_operation(x) = maximum(x[:, 1])
expansive_slice_operation_view(x) = maximum(@view x[:, 1])

for N in [20000, 10000, 100]
	@show N
	R = rand(N,N)

	@btime copied = expansive_slice_operation(R)
	@btime viewed = expansive_slice_operation_view(R)
end

########################################
### Macro expansion
########################################
@macroexpand @. [1, 2, 3] + sin([4, 5, 6])
@macroexpand @show x = 3

########################################
### Metaprogramming
########################################
struct VarNumber{T<:Number}
	value::T
	name::String
end

Base.show(io::IO, x::VarNumber) = print(io,"$(x.name) = $(round(x.value,6))")

for op in [:*, :/, :+, :-, :^, :÷, :%]
	op_str = String(op)

	@eval (Base.$op)(x::VarNumber, y::Number) = VarNumber(($op)(x.value, y), "($(x.name)$($op_str)$(y))")
	@eval (Base.$op)(x::Number, y::VarNumber) = VarNumber(($op)(x, y.value), "($(x)$($op_str)$(y.name))")
	@eval (Base.$op)(x::VarNumber, y::VarNumber) = VarNumber(($op)(x.value, y.value), "($(x.name)$($op_str)$(y.name))")
end

for op in [:sin, :cos, :exp, :log, :tan]
	op_str = String(op)
	@eval (Base.$op)(x::VarNumber) = VarNumber(($op)(x.value), "$($op_str)($(x.name))")
end

a = VarNumber(2.0, "a")
b = VarNumber(3, "b")
c = VarNumber(3.0 + 1.0im, "c")

a^2.0
2.0 * sin(a)  + 3.0
a + 2.0 * exp(c)
a * (b + c)
a * b + c
b % 2

########################################
### automatic derivative
########################################
using Calculus ## finite differences
using ForwardDiff ## automatic differentiation
f(x) = sin(x) / x
df(x) = cos(x) / x - sin(x) / x^2
## correct answer (df(x)/dx) (x = 3.0e-143) = -5.415370496329717*10^126
x = 1.0e-4
df(x)
ForwardDiff.derivative(f, x)
Calculus.derivative(f, x)

x = 1.0e-40
df(x)
ForwardDiff.derivative(f, x)
Calculus.derivative(f, x)


x = 3.0e-143
df(x)
ForwardDiff.derivative(f, x)
Calculus.derivative(f, x)

setprecision(BigFloat, 500)
x = big"1.0e-50000"
df(x)
ForwardDiff.derivative(f, x)
Calculus.derivative(f, x)


g(x) = log(gamma(x))

x = 3.0^-8
digamma(x)
ForwardDiff.derivative(g, x)
Calculus.derivative(g, x)

x = 3.0^-120
digamma(x)
ForwardDiff.derivative(g, x)
# Calculus.derivative(g, x) #fails


########################################
### Scoping
########################################

x,y = 1, 1
function f()
	x = 2
	println("x inside f = ", x)
	println("y inside f = ", y)
end
f()
@show x, y

x = 1
function g()
	global x = 2
	println("x inside g = ", x)
end
g()
@show x

x, y = 1, 1
function h()
	x = 2
	function h_in_h()
		x = 3
		y = 3
	end
	h_in_h()
	println("x inside h = ", x)
	println("y inside f = ", y)
end
h()
@show x, y

### Thanks for Attention
### Stanislav Kazmin (25.05.18)