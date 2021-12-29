pico-8 cartridge // http://www.pico-8.com
version 34
__lua__
-- amphipod
-- by sestrenexsis
-- https://github.com/sestrenexsis/codekatas

-- for advent of code 2021
-- https://adventofcode.com/2021/day/23
_version=1
cartdata("sestrenexsis_amphipod_1")

--[[ save data
 0: lowest score, red
 1: lowest score, orange
 2: lowest score, yellow
 3: lowest score, green
--]]
-->8
-- tile

tile={}

function tile:new(r,c,t)
	local obj={
		row=r,
		col=c,
		typ=t,
	}
	return setmetatable(
		obj,{__index=self}
	)
end

-- board

board={}

function board:new(depth)
	local tiles={}
	for c=2,12 do
		if (
			c==4 or
			c==6 or
			c==8 or
			c==10
		) then
			add(tiles,tile:new(5,c,5))
		else
			add(tiles,tile:new(5,c,6))
		end
	end
	for r=1,depth do
		add(tiles,tile:new(5+r,4,1))
		add(tiles,tile:new(5+r,6,2))
		add(tiles,tile:new(5+r,8,3))
		add(tiles,tile:new(5+r,10,4))
	end
	local obj={
		tiles=tiles,
	}
	return setmetatable(
		obj,{__index=self}
	)
end
-->8
-- helper functions

function cleanmap()
	-- clear map
	for y=3,13 do
		for x=1,14 do
			mset(x,y,0)
		end
	end
	-- process tiles
	for tile in all(_bd.tiles) do
		for dx=-1,1 do
			for dy=-1,1 do
				local x=tile.col+dx
				local y=tile.row+dy
				if dx==0 and dy==0 then
					mset(x,y,tile.typ)
				elseif mget(x,y)==0 then
					local typ=56
					if y<8 then typ=56
					elseif x==4 then typ=57
					elseif x==6 then typ=58
					elseif x==8 then typ=59
					elseif x==10 then typ=60
					end
					mset(x,y,typ)
				end
			end
		end
	end
end
-->8
-- 

function _init()
	_bd=board:new(2)
end

function _update()
	local dirty=false
	if btnp(❎) then
		_bd=board:new(2)
		dirty=true
	end
	if btnp(🅾️) then
		_bd=board:new(4)
		dirty=true
	end
	if dirty then cleanmap() end
end

function _draw()
	cls()
	map(0,0,0,0,128,128)
	for tile in all(_bd.tiles) do
		local lft=8*tile.col
		local top=8*tile.row
		--rect(lft,top,lft+7,top+7,8)
	end
end
__gfx__
00000000111111113333333399999999888888885555555555555555770000770000000000000000000000000000000000000000002222222222222222222200
00000000111111113333333399999999888888885555555555555555700000070770077000cccc0000bbbb0000aaaa0000eeee00020000000000000000000020
00700700111cc111333bb333999aa999888ee888555555555555555500000000070000700cccccc00bbbbbb00aaaaaa00eeeeee0200222222222222222222002
0007700011c11c1133b33b3399a99a9988e88e88555555555556655500000000000000000cccccc00bbbbbb00aaaaaa00eeeeee0202000000000000000000202
0007700011c11c1133b33b3399a99a9988e88e88555555555556655500000000000000000cccccc00bbbbbb00aaaaaa00eeeeee0202000000000000000000202
00700700111cc111333bb333999aa999888ee888555555555555555500000000070000700cccccc00bbbbbb00aaaaaa00eeeeee0202000000000000000000202
00000000111111113333333399999999888888885555555555555555700000070770077000cccc0000bbbb0000aaaa0000eeee00202000000000000000000202
00000000111111113333333399999999888888885555555555555555770000770000000000000000000000000000000000000000202000000000000000000202
0000000000000000000000000000000000000000000000000000000000000000000000000cccc0000bbbb0000aaaa0000eeee000202000000000000000000202
005555000055550000555500055005500555555000555500005555000555550000000000cccccc00bbbbbb00aaaaaa00eeeeee00202000000000000000000202
055555500555555005555550055005500555555005555550055555500555555000000000cccccc00bbbbbb00aaaaaa00eeeeee00202000000000000000000202
055555500555555005555550055005500005500005555550055555500555555000000000cccccc00bbbbbb00aaaaaa00eeeeee00202000000000000000000202
055005500550505005500550055005500005500005500550055005500550055000000000cccccc00bbbbbb00aaaaaa00eeeeee00202000000000000000000202
0550055005505050055005500550055000055000055005500550055005500550000000000cccc0000bbbb0000aaaa0000eeee000202000000000000000000202
05500550055050500550055005500550000550000550055005500550055005500000000000000000000000000000000000000000202000000000000000000202
05555550055050500555555005555550000550000555555005500550055005500000000000000000000000000000000000000000202000000000000000000202
05555550055050500555550005555550000550000555550005500550055005500000000000000000000000000000000000000000202000000000000000000202
05500550055050500550000005500550000550000550000005500550055005500000000000000000000000000000000000000000202000000000000000000202
05500550055050500550000005500550000550000550000005500550055005500000000000000000000000000000000000000000202000000000000000000202
05500550055050500550000005500550000550000550000005500550055005500000000000000000000000000000000000000000202000000000000000000202
05500550055050500550000005500550000550000550000005500550055005500000000000000000000000000000000000000000202000000000000000000202
05500550055050500550000005500550055555500550000005555550055555500000000000000000000000000000000000000000200222222222222222222002
05500550055050500550000005500550055555500550000000555500055555000000000000000000000000000000000000000000020000000000000000000020
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002222222222222222222200
0000000000000000000000000000000000000000000000000000000000000000777777767777777c7777777b7777777a7777777e000000000000000000000000
0000000000000000000000000000000000000000000000000000000000000000766666657cccccc17bbbbbb37aaaaaa97eeeeee8000000000000000000000000
0000000000000000000000000000000000000000000000000000000000000000766666657cccccc17bbbbbb37aaaaaa97eeeeee8000000000000000000000000
0000000000000000000000000000000000000000000000000000000000000000766766657cc7ccc17bb7bbb37aa7aaa97ee7eee8000000000000000000000000
0000000000000000000000000000000000000000000000000000000000000000766656657ccc3cc17bbb3bb37aaa9aa97eee8ee8000000000000000000000000
0000000000000000000000000000000000000000000000000000000000000000766666657cccccc17bbbbbb37aaaaaa97eeeeee8000000000000000000000000
0000000000000000000000000000000000000000000000000000000000000000766666657cccccc17bbbbbb37aaaaaa97eeeeee8000000000000000000000000
000000000000000000000000000000000000000000000000000000000000000065555555c1111111b3333333a9999999e8888888000000000000000000000000
__label__
00222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222200
02000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000020
20022222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222002
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000005555000055550000555500055005500555555000555500005555000555550000000000000000000000000000000202
20200000000000000000000000000000055555500555555005555550055005500555555005555550055555500555555000000000000000000000000000000202
20200000000000000000000000000000055555500555555005555550055005500005500005555550055555500555555000000000000000000000000000000202
20200000000000000000000000000000055005500550505005500550055005500005500005500550055005500550055000000000000000000000000000000202
20200000000000000000000000000000055005500550505005500550055005500005500005500550055005500550055000000000000000000000000000000202
20200000000000000000000000000000055005500550505005500550055005500005500005500550055005500550055000000000000000000000000000000202
20200000000000000000000000000000055555500550505005555550055555500005500005555550055005500550055000000000000000000000000000000202
20200000000000000000000000000000055555500550505005555500055555500005500005555500055005500550055000000000000000000000000000000202
20200000000000000000000000000000055005500550505005500000055005500005500005500000055005500550055000000000000000000000000000000202
20200000000000000000000000000000055005500550505005500000055005500005500005500000055005500550055000000000000000000000000000000202
20200000000000000000000000000000055005500550505005500000055005500005500005500000055005500550055000000000000000000000000000000202
20200000000000000000000000000000055005500550505005500000055005500005500005500000055005500550055000000000000000000000000000000202
20200000000000000000000000000000055005500550505005500000055005500555555005500000055555500555555000000000000000000000000000000202
20200000000000000000000000000000055005500550505005500000055005500555555005500000005555000555550000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000777777767777777677777776777777767777777677777776777777767777777677777776777777767777777677777776777777760000000000000202
20200000766666657666666576666665766666657666666576666665766666657666666576666665766666657666666576666665766666650000000000000202
20200000766666657666666576666665766666657666666576666665766666657666666576666665766666657666666576666665766666650000000000000202
20200000766766657667666576676665766766657667666576676665766766657667666576676665766766657667666576676665766766650000000000000202
20200000766656657666566576665665766656657666566576665665766656657666566576665665766656657666566576665665766656650000000000000202
20200000766666657666666576666665766666657666666576666665766666657666666576666665766666657666666576666665766666650000000000000202
20200000766666657666666576666665766666657666666576666665766666657666666576666665766666657666666576666665766666650000000000000202
20200000655555556555555565555555655555556555555565555555655555556555555565555555655555556555555565555555655555550000000000000202
20200000777777765555555555555555555555555555555555555555555555555555555555555555555555555555555555555555777777760000000000000202
20200000766666655555555555555555555555555555555555555555555555555555555555555555555555555555555555555555766666650000000000000202
20200000766666655555555555555555555555555555555555555555555555555555555555555555555555555555555555555555766666650000000000000202
20200000766766655556655555566555555665555556655555566555555665555556655555566555555665555556655555566555766766650000000000000202
20200000766656655556655555566555555665555556655555566555555665555556655555566555555665555556655555566555766656650000000000000202
20200000766666655555555555555555555555555555555555555555555555555555555555555555555555555555555555555555766666650000000000000202
20200000766666655555555555555555555555555555555555555555555555555555555555555555555555555555555555555555766666650000000000000202
20200000655555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555655555550000000000000202
20200000777777767777777677777776555555557777777655555555777777765555555577777776555555557777777677777776777777760000000000000202
20200000766666657666666576666665555555557666666555555555766666655555555576666665555555557666666576666665766666650000000000000202
20200000766666657666666576666665555555557666666555555555766666655555555576666665555555557666666576666665766666650000000000000202
20200000766766657667666576676665555665557667666555566555766766655556655576676665555665557667666576676665766766650000000000000202
20200000766656657666566576665665555665557666566555566555766656655556655576665665555665557666566576665665766656650000000000000202
20200000766666657666666576666665555555557666666555555555766666655555555576666665555555557666666576666665766666650000000000000202
20200000766666657666666576666665555555557666666555555555766666655555555576666665555555557666666576666665766666650000000000000202
20200000655555556555555565555555555555556555555555555555655555555555555565555555555555556555555565555555655555550000000000000202
20200000000000000000000077777776555555557777777655555555777777765555555577777776555555557777777600000000000000000000000000000202
20200000000000000000000076666665555555557666666555555555766666655555555576666665555555557666666500000000000000000000000000000202
20200000000000000000000076666665555555557666666555555555766666655555555576666665555555557666666500000000000000000000000000000202
20200000000000000000000076676665555665557667666555566555766766655556655576676665555665557667666500000000000000000000000000000202
20200000000000000000000076665665555665557666566555566555766656655556655576665665555665557666566500000000000000000000000000000202
20200000000000000000000076666665555555557666666555555555766666655555555576666665555555557666666500000000000000000000000000000202
20200000000000000000000076666665555555557666666555555555766666655555555576666665555555557666666500000000000000000000000000000202
20200000000000000000000065555555555555556555555555555555655555555555555565555555555555556555555500000000000000000000000000000202
20200000000000000000000077777776777777767777777677777776777777767777777677777776777777767777777600000000000000000000000000000202
20200000000000000000000076666665766666657666666576666665766666657666666576666665766666657666666500000000000000000000000000000202
20200000000000000000000076666665766666657666666576666665766666657666666576666665766666657666666500000000000000000000000000000202
20200000000000000000000076676665766766657667666576676665766766657667666576676665766766657667666500000000000000000000000000000202
20200000000000000000000076665665766656657666566576665665766656657666566576665665766656657666566500000000000000000000000000000202
20200000000000000000000076666665766666657666666576666665766666657666666576666665766666657666666500000000000000000000000000000202
20200000000000000000000076666665766666657666666576666665766666657666666576666665766666657666666500000000000000000000000000000202
20200000000000000000000065555555655555556555555565555555655555556555555565555555655555556555555500000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000202
20022222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222002
02000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000020
00222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222200
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

__map__
0d0e0e0e0e0e0e0e0e0e0e0e0e0e0e0f0d0e0e0e0e0e0e0e0e0e0e0e0e0e0e0f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000010111213141516170000001f1d00101112131415161700141400001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d30313120212223242526270000001f1d30202122232425262700242400001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f1d00000000000000000000000000001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f1d01010101010101010101010101001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f1d01020202020202020202020201001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f1d01010105010401050103010101001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f1d00000103010401050106010000001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000039003a003b003c000000001f1d00000103010501060104010000001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f1d00000106010301040106010000001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f1d0000011c011b011a0119010000001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f1d00000000000000000000000000001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f1d00000000000000000000000000001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f1d00000000000000000000000000001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
2d2e2e2e2e2e2e2e2e2e2e2e2e2e2e2f2d2e2e2e2e2e2e2e2e2e2e2e2e2e2e2f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
0d0e0e0e0e0e0e0e0e0e0e0e0e0e0e0f00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000010111213141516170000001f00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d30313120212223242526270000001f00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d01010101010101010101010101001f00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d01020202020202020202020201001f00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d01010105010401050103010101001f00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000106010301040106010000001f00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d0000011c011b011a0119010000001f00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
2d2e2e2e2e2e2e2e2e2e2e2e2e2e2e2f00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000