module UnfoldsTests

using ..Unfolds: Unfolds, unfold
using Test

@testset "unfold" begin
    @testset "eltype" begin
        @test eltype(unfold(x -> nothing, 1; eltype=String)) == String

        function fib_int(x)
            Iterators.take(unfold((1, 1); eltype=Int) do (a, b)
                a, (b, a+b)
            end, x)
        end

        @test eltype(fib_int(1000)) == Int
        @test eltype(collect(fib_int(4))) == Int
        @test collect(fib_int(4)) == [1, 1, 2, 3]
    end

    @testset "size" begin
        bad_one_to(n, size) = Unfolds.unfold(x -> x > n ? nothing : (x, x+1), 1; size)
        @test Base.IteratorSize(bad_one_to(10, 10)) == Base.HasLength()
        @test Base.IteratorSize(bad_one_to(10, (10,))) == Base.HasShape{1}()
        @test Base.IteratorSize(bad_one_to(10, Base.SizeUnknown())) == Base.SizeUnknown()
        @test collect(bad_one_to(10, 10)) == 1:10
        @test collect(bad_one_to(10, (10,))) == 1:10
        @test collect(bad_one_to(10, Base.SizeUnknown())) == 1:10

        infinite_itr = Unfolds.unfold(x -> (x, x), 1; size=Base.IsInfinite())
        @test Base.IteratorSize(infinite_itr) == Base.IsInfinite()
        # collect refuses to try and collect iterators of infinite size
        @test_throws MethodError collect(infinite_itr)

        shaped_itr1 = bad_one_to(9, (3, 3))
        @test collect(shaped_itr1) == reshape(1:9, (3, 3))
    end

    @testset "size and eltype" begin
        itr1 = Unfolds.unfold(x -> x > 9 ? nothing : (x, x+1), 1; eltype=Int, size=9)
        @test collect(itr1) == 1:9

        itr2 = Unfolds.unfold(x -> x > 9 ? nothing : (x, x+1), 1; eltype=Int, size=(3, 3))
        @test collect(itr2) == reshape(1:9, (3, 3))
    end
end

end
