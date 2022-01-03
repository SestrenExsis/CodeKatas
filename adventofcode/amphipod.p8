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
-- classes

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
	add(tiles,tile:new(5,2,6))
	add(tiles,tile:new(5,3,6))
	add(tiles,tile:new(5,4,5))
	add(tiles,tile:new(5,5,6))
	add(tiles,tile:new(5,6,5))
	add(tiles,tile:new(5,7,6))
	add(tiles,tile:new(5,8,5))
	add(tiles,tile:new(5,9,6))
	add(tiles,tile:new(5,10,5))
	add(tiles,tile:new(5,11,6))
	add(tiles,tile:new(5,12,6))
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

-- amphipods
amphipod={}

function amphipod:new(r,c,t)
	local obj={
		row=r,
		col=c,
		typ=t,
	}
	return setmetatable(
		obj,{__index=self}
	)
end

-- move
move={}

function move:new(r,c)
	local obj={
		row=r,
		col=c,
		amf=nil,
		path={},
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
	for tile in all(_brd.tiles) do
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
	_dirty=false
end

function getamphipods(depth)
	local res={}
	for d=1,depth do
		add(res,amphipod:new(5+d,4,1))
		add(res,amphipod:new(5+d,6,2))
		add(res,amphipod:new(5+d,8,3))
		add(res,amphipod:new(5+d,10,4))
	end
	return res
end
-->8
-- main

function _init()
	_mov=move:new(5,7)
	_brd=board:new(2)
	_amfs=getamphipods(2)
	_dirty=true
end

function _update()
	-- check for movement input
	local ncol=_mov.col
	local nrow=_mov.row
	if btnp(⬅️) then
		ncol-=1
	elseif btnp(➡️) then
		ncol+=1
	end
	if btnp(⬆️) then
		nrow-=1
	elseif btnp(⬇️) then
		nrow+=1
	end
	-- check for valid tile
	local valid=false
	for tile in all(_brd.tiles) do
		if (
			tile.col==ncol and
			tile.row==nrow
		) then
			valid=true
			break
		end
	end
	if valid and
	(
		_mov.col!=ncol or
		_mov.row!=nrow
	) then
		-- check if backtracking
		local backtrack=false
		for dot in all(_mov.path) do
			if (
				dot[1]==ncol and
				dot[2]==nrow
			) then
				backtrack=true
			end
		end
		if backtrack then
			-- delete prev dot
			for i=1,#_mov.path do
				local dot=_mov.path[i]
				if (
					dot[1]==ncol and
					dot[2]==nrow
				) then
					deli(_mov.path,i)
					break
				end
			end
		elseif _mov.amf!=nil then
			-- add dot
			add(_mov.path,
				{_mov.col,_mov.row}
			)
		end
		_mov.col=ncol
		_mov.row=nrow
	end
	-- check for grabbed amphipod
	if btnp(❎) then
		if _mov.amf==nil then
			for amf in all(_amfs) do
				if (
					amf.col==_mov.col and
					amf.row==_mov.row
				) then
					_mov.amf=amf
					break
				end
			end
		else
			_mov.amf.col=_mov.col
			_mov.amf.row=_mov.row
			_mov.amf=nil
		end
	end
	-- recreate board
	if btnp(🅾️) then
		if rnd()<0.5 then
			_mov=move:new(5,7)
			_brd=board:new(2)
			_amfs=getamphipods(2)
		else
			_mov=move:new(5,7)
			_brd=board:new(4)
			_amfs=getamphipods(4)
		end
		_dirty=true
	end
	if _dirty then cleanmap() end
end

function _draw()
	cls()
	map(0,0,0,0,128,128)
	-- draw movement line
	local dist=0
	if _mov.amf!=nil then
		dist=(
			abs(_mov.col-_mov.amf.col)+
			abs(_mov.row-_mov.amf.row)
		)
		local typ=_mov.amf.typ
		local cs={12,11,10,14}
		local c=cs[typ]
		local sx=8*_mov.amf.col+3
		local sy=8*_mov.amf.row+3
		local ex=8*_mov.col+3
		local ey=8*_mov.row+3
		line(sx,sy,sx,ey,2)
		line(sx,ey,ex,ey,2)
		line(sx+1,sy,sx+1,ey,2)
		line(sx+1,ey,ex+1,ey,2)
		line(sx,sy+1,sx,ey+1,2)
		line(sx,ey+1,ex,ey+1,2)
		line(sx+1,sy+1,sx+1,ey+1,2)
		line(sx+1,ey+1,ex+1,ey+1,2)
	end
	-- draw amphipods
	for amf in all(_amfs) do
		local lft=8*amf.col
		local top=8*amf.row
		local fm=8+amf.typ
		if amf==_mov.amf then
			lft=8*_mov.col
			top=8*_mov.row
			fm=24+amf.typ
		end
		spr(fm,lft,top)
	end
	local fm=7
	if btn(❎) then
		fm=8
	end
	spr(fm,8*_mov.col,8*_mov.row)
	-- draw debug
	print(dist,4,4)
	for dot in all(_mov.path) do
		spr(40,8*dot[1],8*dot[2])
	end
end
__gfx__
00000000111111113333333399999999888888885555555555555555770000770000000000000000000000000000000000000000002222222222222222222200
00000000111111113333333399999999888888885555555555555555700000070770077000cccc0000bbbb0000aaaa0000eeee00020000000000000000000020
00700700111cc111333bb333999aa999888ee888555555555555555500000000070000700cc7ccc00bb7bbb00aa7aaa00ee7eee0200222222222222222222002
0007700011c11c1133b33b3399a99a9988e88e88555555555556655500000000000000000c7cccc00b7bbbb00a7aaaa00e7eeee0202000000000000000000202
0007700011c11c1133b33b3399a99a9988e88e88555555555556655500000000000000000cccccc00bbbbbb00aaaaaa00eeeeee0202000000000000000000202
00700700111cc111333bb333999aa999888ee888555555555555555500000000070000700cccccc00bbbbbb00aaaaaa00eeeeee0202000000000000000000202
00000000111111113333333399999999888888885555555555555555700000070770077000cccc0000bbbb0000aaaa0000eeee00202000000000000000000202
00000000111111113333333399999999888888885555555555555555770000770000000000000000000000000000000000000000202000000000000000000202
0000000000000000000000000000000000000000000000000000000000000000000000000cccc0000bbbb0000aaaa0000eeee000202000000000000000000202
005555000055550000555500055005500555555000555500005555000555550000000000cc7ccc00bb7bbb00aa7aaa00ee7eee00202000000000000000000202
055555500555555005555550055005500555555005555550055555500555555000000000c7cccc00b7bbbb00a7aaaa00e7eeee00202000000000000000000202
055555500555555005555550055005500005500005555550055555500555555000000000cccccc00bbbbbb00aaaaaa00eeeeee00202000000000000000000202
055005500550505005500550055005500005500005500550055005500550055000000000cccccc00bbbbbb00aaaaaa00eeeeee00202000000000000000000202
0550055005505050055005500550055000055000055005500550055005500550000000000cccc0000bbbb0000aaaa0000eeee000202000000000000000000202
05500550055050500550055005500550000550000550055005500550055005500000000000000000000000000000000000000000202000000000000000000202
05555550055050500555555005555550000550000555555005500550055005500000000000000000000000000000000000000000202000000000000000000202
05555550055050500555550005555550000550000555550005500550055005500000000000000000000000000000000000000000202000000000000000000202
05500550055050500550000005500550000550000550000005500550055005500000000000000000000000000000000000000000202000000000000000000202
055005500550505005500000055005500005500005500000055005500550055000066000000cc000000bb000000aa000000ee000202000000000000000000202
05500550055050500550000005500550000550000550000005500550055005500067660000c7cc0000b7bb0000a7aa0000e7ee00202000000000000000000202
05500550055050500550000005500550000550000550000005500550055005500066660000cccc0000bbbb0000aaaa0000eeee00202000000000000000000202
055005500550505005500000055005500555555005500000055555500555555000066000000cc000000bb000000aa000000ee000200222222222222222222002
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
1d00000020212223242526270000001f1d30202122232425262700242400001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f1d00000000000000000000000000001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f1d01010101010101010101010101001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f1d01020202020202020202020201001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f1d01010105010401050103010101001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f1d00000103010401050106010000001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f1d00000103010501060104010000001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
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
