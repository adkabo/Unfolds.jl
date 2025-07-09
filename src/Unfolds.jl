# This code was written by Colin Caine (@cmcaine).
# https://github.com/JuliaLang/julia/pull/44873#issuecomment-1567192786

module Unfolds

export unfold

using Base:
    SizeUnknown, HasLength, HasShape, IsInfinite, EltypeUnknown, HasEltype,
    @propagate_inbounds

import Base:
    length, size, eltype, IteratorSize, IteratorEltype, iterate, isdone

const SizeTypes = Union{SizeUnknown, IsInfinite, <:Integer, <:NTuple{N, <:Integer} where {N}}
size_type_to_iteratorsize(T::Type{<:Union{SizeUnknown, IsInfinite}}) = T()
size_type_to_iteratorsize(::Type{<:Integer}) = HasLength()
size_type_to_iteratorsize(::Type{<:NTuple{N, <:Integer}}) where {N} = HasShape{N}()

"""
    unfold(f, initialstate; [eltype], [size])

Iterable object that generates values from an initial state and a transition
function `f(state)`. The function must follow the same rules as `iterate`.
It returns either `(newvalue, newstate)` or `nothing`, in which case the
sequence ends.

The optional parameters `eltype` and `size` specify the element type and size of the iterator.

If `size` is specified it must be one of:

- an integer, representing the length of the iterator
- a tuple of integers, representing the `size` of the iterator (length will be defined as `prod(size)`)
- `Base.IsInfinite()`, meaning that the iterator is of infinite length
- `Base.SizeUnknown()`, if the iterator has an unknown length (this is the default).

See also: [`iterate`](@ref), [the iteration interface](@ref man-interface-iteration)

!!! compat "Julia 1.10"
    This function was added in Julia 1.10.

# Examples

```jldoctest
julia> fib = Iterators.unfold((1,1)) do (a,b)
           a, (b, a+b)
       end;

julia> reduce(hcat, Iterators.take(fib, 7))
1×7 Matrix{Int64}:
 1  1  2  3  5  8  13

julia> frac(c, z=0.0im) = Iterators.unfold((c, z); eltype=ComplexF64) do (c, z)
           if real(z * z') < 4
               z, (c, z^2 + c)
           else
               nothing
           end
       end;

julia> [count(Returns(true), frac(-0.835-0.2321im, (k+j*im)/6)) for j in -4:4, k in -8:8]
9×17 Matrix{Int64}:
  2   2   2   3   3   3   5  41   8   4   3   3   2   2   2   2   1
  2   3   5   4   5   8  20  11  17  23   4   3   3   3   2   2   2
  4  10  17  12   7  56  18  58  33  22   6   5   4   5   4   3   2
 26  56  15  13  18  23  13  14  27  46   8   9  16  12   8   4   3
 10   7  62  17  16  23  11  12  39  12  11  23  16  17  62   7  10
  3   4   8  12  16   9   8  46  27  14  13  23  18  13  15  56  26
  2   3   4   5   4   5   6  22  33  58  18  56   7  12  17  10   4
  2   2   2   3   3   3   4  23  17  11  20   8   5   4   5   3   2
  1   2   2   2   2   3   3   4   8  41   5   3   3   3   2   2   2
```

# Extended help

The interface for `f` is very similar to the interface required by `iterate`, but `unfold` is simpler to use because it does not require you to define a type. You can use this to your advantage when prototyping or writing one-off iterators.

You may want to define an iterator type instead for readability or to dispatch on the type of your iterator.

`unfold` is related to a `while` loop because:
```julia
collect(unfold(f, initialstate))
```
is roughly the same as:
```julia
acc = []
state = initialstate
while true
    x = f(state)
    isnothing(x) && break
    element, state = x
    push!(acc, element)
end
```
But the `unfold` version may produce a more strictly typed vector and can be easily modified to return a lazy collection by removing `collect()`.

In Haskell and some other functional programming environments, this function is known as `unfoldr`.
"""
function unfold(f, initialstate; eltype=nothing, size::SizeTypes=SizeUnknown())
    rest(Unfold(f, eltype), initialstate; size)
end

"""
    unfold(f; [eltype], [size])

Create a function that will return an iterator unfolded by `f` when given an initial state. Equivalent to `initial -> unfold(f, initial; eltype, size)`.

# Example

```jldoctest
julia> const collatz_path = Iterators.unfold() do n
           if isnothing(n)
               n
           elseif isone(n)
               (n, nothing)
           else
               (n, iseven(n) ? n÷2 : 3n+1)
           end
       end
#1 (generic function with 1 method)

julia> collatz_path(3) |> collect
8-element Vector{Int64}:
  3
 10
  5
 16
  8
  4
  2
  1
```
"""
function unfold(f; eltype=nothing, size::SizeTypes=SizeUnknown())
    initial -> unfold(f, initial; eltype, size)
end

struct Unfold{Eltype, FuncType}
    f::FuncType

    Unfold{E, F}(f::F) where {E, F} = new{E, F}(f)
    Unfold(f::F, eltype) where {F} = new{eltype, F}(f)
end
Unfold(f) = Unfold(f, nothing)

eltype(::Type{<:Unfold{Eltype}}) where {Eltype} = Eltype
eltype(::Type{<:Unfold{nothing}}) = Any
IteratorEltype(::Type{<:Unfold{nothing}}) = EltypeUnknown()
IteratorEltype(::Type{<:Unfold}) = HasEltype()

IteratorSize(::Type{<:Unfold}) = SizeUnknown()

@propagate_inbounds iterate(it::Unfold, state) = it.f(state)

# Iterators.Rest, but it can know how big the iterator will be.
struct Rest{I,S,Z<:SizeTypes}
    itr::I
    st::S
    size::Z
end

"""
    rest(iter, state; [size])

An iterator that yields the same elements as `iter`, but starting at the given `state`.

If `size` is specified it must be one of:

- an integer, representing the length of the returned iterator
- a tuple of integers, representing the `size` of the returned iterator (length will be defined as `prod(size)`)
- `Base.IsInfinite()`, meaning that the returned iterator is of infinite length
- `Base.SizeUnknown()`, if the returned iterator has an unknown length

!!! compat "Julia 1.10"
    The `size` parameter was added in Julia 1.10.

See also: [`Iterators.drop`](@ref), [`Iterators.peel`](@ref), [`Base.rest`](@ref).

# Examples
```jldoctest
julia> collect(Iterators.rest([1,2,3,4], 2))
3-element Vector{Int64}:
 2
 3
 4
```
"""
rest(itr, state; size=rest_iteratorsize(itr)) = Rest(itr, state, size)
rest(itr::Rest, state; size=rest_iteratorsize(itr)) = Rest(itr.itr, state, size)
rest(itr) = itr

@propagate_inbounds iterate(i::Rest, st=i.st) = iterate(i.itr, st)
isdone(i::Rest, st...) = isdone(i.itr, st...)

eltype(::Type{<:Rest{I}}) where {I} = eltype(I)
IteratorEltype(::Type{<:Rest{I}}) where {I} = IteratorEltype(I)

rest_iteratorsize(a) = SizeUnknown()
rest_iteratorsize(::IsInfinite) = IsInfinite()

IteratorSize(::Type{<:Rest{<:Any, <:Any, Z}}) where {Z} = size_type_to_iteratorsize(Z)
length(u::Rest{<:Any, <:Any, <:Integer}) = u.size
size(u::Rest{<:Any, <:Any, <:NTuple{N, <:Integer}}) where {N} = u.size
length(u::Rest{<:Any, <:Any, <:NTuple{N, <:Integer}}) where {N} = prod(u.size)

end
