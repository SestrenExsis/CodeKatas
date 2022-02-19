pico-8 cartridge // http://www.pico-8.com
version 35
__lua__
-- amphipod
-- by sestrenexsis
-- https://github.com/sestrenexsis/codekatas

-- for advent of code 2021
-- https://adventofcode.com/2021/day/23
_version=1
cartdata("sestrenexsis_amphipod_1")

--[[ save data
 0: lowest score
--]]
-->8
-- helper functions

function cleanmap()
	-- clear map
	for y=3,13 do
		for x=1,14 do
			mset(x,y,0)
		end
	end
	-- add walls
	for x0=1,#_cels do
		for dx=-1,1 do
			for dy=-1,1 do
				mset(x0+dx+1,5+dy,56)
			end
		end
	end
	-- add main hallway
	for x0=1,#_cels do
		mset(x0+1,5,6)
	end
	-- 
	for typ=1,4 do
		local x=2*typ+2
		mset(x,5,5)
		for y=1,_size do
			mset(x-1,5+y,56)
			mset(x  ,5+y,typ)
			mset(x+1,5+y,56)
		end
		mset(x-1,6+_size,56)
		mset(x  ,6+_size,56+typ)
		mset(x+1,6+_size,56)
	end
	--[[
	if (
		x==3 or
		x==5 or
		x==7 or
		x==9
	--]]
	_dirty=false
end
-->8
-- main

function restart()
	_move={}
	_costs={0}
	_lx=7
	_x=7
	_amf=nil
	_cels={}
	for col=1,11 do
		add(_cels,{})
	end
	-- make a bag of amphipods
	local bag={}
	for d=1,_size do
		for i=1,4 do
			add(bag,i)
		end
	end
	-- randomly assign amphipods
	for col=3,9,2 do
		for i=1,_size do
			local idx=1+flr(rnd(#bag))
			local amf=bag[idx]
			add(_cels[col],amf)
			bag[idx]=bag[#bag]
			deli(bag,#bag)
		end
	end
	cleanmap()
end

function restart2()
	menuitem(1,
		"set depth to 4",
		restart4
	)
	_size=2
	restart()
end

function restart4()
	menuitem(1,
		"set depth to 2",
		restart2
	)
	_size=4
	restart()
end

function _init()
	restart2()
end

function _update()
	-- check for input
	local lx=_x
	if btnp(⬅️) then
		_x-=1
	elseif btnp(➡️) then
		_x+=1
	end
	_x=mid(1,_x,#_cels)
	local cap=0
	if (
		_x==3 or
		_x==5 or
		_x==7 or
		_x==9
	) then
		cap=_size
	end
	-- check for amf collision
	if (
		_amf!=nil and
		#_cels[_x]>cap
	) then
		_x=lx
	end
	-- grab/drop amphipod
	if btnp(❎) or btnp(⬇️) then
		if _amf==nil then
			-- grab the top amphipod
			if #_cels[_x]>0 then
				_amf=_cels[_x][#_cels[_x]]
				deli(_cels[_x],#_cels[_x])
				_lx=_x
			end
		else
			-- drop held amphipod
			add(_cels[_x],_amf)
			_amf=nil
			if _lx!=_x then
				add(_move,{_lx,_x})
				add(_costs,0)
			end
		end
	elseif btnp(🅾️) then
		-- undo last move
		if _amf==nil then
			if #_move>0 then
				local mov=_move[#_move]
				_x=mov[1]
				_lx=_x
				local id=mov[2]
				_amf=_cels[id][#_cels[id]]
				deli(_cels[id],#_cels[id])
				deli(_move,#_move)
				deli(_costs,#_costs)
			elseif #_costs>0 then
				deli(_costs,#_costs)
			else
				_costs[#_costs]=0
			end
		else
			add(_cels[_lx],_amf)
			add(_costs,0)
			_amf=nil
		end
	end
	-- calculate current cost
	local cost=0
	if _amf!=nil then
		-- add grab cost
		if (
			_lx==3 or
			_lx==5 or
			_lx==7 or
			_lx==9
		) then
			cost+=_size-#_cels[_lx]
		end
		-- add travel cost
		local xmin=min(_x,_lx)
		local xmax=max(_x,_lx)
		cost+=xmax-xmin
		-- add drop cost
		if _x!=_lx and (
			_x==3 or
			_x==5 or
			_x==7 or
			_x==9
		) then
			cost+=_size-#_cels[_x]
		end
	end
	if _amf!=nil then
		cost*=10^(_amf-1)
	end
	_costs[#_costs]=cost
end

function _draw()
	cls()
	map(0,0,0,0,128,128)
	-- draw movement dots
	if _amf!=nil then
		-- draw drop dots
		if (
			_x==3 or
			_x==5 or
			_x==7 or
			_x==9
		) then
			for i=1,_size-#_cels[_x] do
				local lft=8*(1+_x)
				local top=8*(5+i)
				if t()%0.5<0.25 then
					spr(48+_amf,lft,top)
				end
			end
		end
		-- draw grab dots
		if (
			_lx==3 or
			_lx==5 or
			_lx==7 or
			_lx==9
		) then
			for i=1,_size-#_cels[_lx] do
				local lft=8*(1+_lx)
				local top=8*(5+i)
				spr(40+_amf,lft,top)
			end
		end
		-- draw travel dots
		local xmin=min(_x,_lx)
		local xmax=max(_x,_lx)
		for i=xmin,xmax do
			local lft=8*(1+i)
			local top=8*(5)
			spr(40+_amf,lft,top)
		end
	end
	-- draw amphipods
	palt(0,false)
	palt(15,true)
	for x=1,#_cels do
		local ht=0
		if (
			x==3 or
			x==5 or
			x==7 or
			x==9
		) then
			ht=_size
		end
		local lft=8*(1+x)
		for y=1,#_cels[x] do
			local top=8*(6+ht-y)
			local amf=_cels[x][y]
			spr(8+amf,lft,top)
		end
	end
	-- draw grabbed amphipod
	if _amf!=nil then
		spr(24+_amf,8*(_x+1),8*5)
	end
	palt()
	-- draw cursor
	local fm=7
	if btn(❎) then
		fm=8
	end
	local y0=5
	spr(fm,8*(1+_x),8*5)
	-- draw costs
	for i=1,#_costs do
		print(_costs[i],4,4+6*(i-1),15)
	end
end
__gfx__
000000001111111133333333999999998888888855555555555555557700007700000000ffffffffffffffffffffffffffffffff002222222222222222222200
000000001111111133333333999999998888888855555555555555557000000707700770ffccccffffbbbbffffaaaaffffeeeeff020000000000000000000020
007007001111111133333333999999998888888855555555555555550000000007000070fcc7cccffbb7bbbffaa7aaaffee7eeef200222222222222222222002
00077000111cc111333bb333999aa999888ee88855555555555665550000000000000000fc7ccccffb7bbbbffa7aaaaffe7eeeef202000000000000000000202
00077000111cc111333bb333999aa999888ee88855555555555665550000000000000000fccccccffbbbbbbffaaaaaaffeeeeeef202000000000000000000202
007007001111111133333333999999998888888855555555555555550000000007000070fccccccffbbbbbbffaaaaaaffeeeeeef202000000000000000000202
000000001111111133333333999999998888888855555555555555557000000707700770ffccccffffbbbbffffaaaaffffeeeeff202000000000000000000202
000000001111111133333333999999998888888855555555555555557700007700000000ffffffffffffffffffffffffffffffff202000000000000000000202
000000000000000000000000000000000000000000000000000000000000000000000000fccccffffbbbbffffaaaaffffeeeefff202000000000000000000202
005555000055550000555500055005500555555000555500005555000555550000000000cc7cccffbb7bbbffaa7aaaffee7eeeff202000000000000000000202
055555500555555005555550055005500555555005555550055555500555555000000000c7cccc0fb7bbbb0fa7aaaa0fe7eeee0f202000000000000000000202
055555500555555005555550055005500005500005555550055555500555555000000000cccccc0fbbbbbb0faaaaaa0feeeeee0f202000000000000000000202
055005500550505005500550055005500005500005500550055005500550055000000000cccccc0fbbbbbb0faaaaaa0feeeeee0f202000000000000000000202
055005500550505005500550055005500005500005500550055005500550055000000000fcccc00ffbbbb00ffaaaa00ffeeee00f202000000000000000000202
055005500550505005500550055005500005500005500550055005500550055000000000ff0000ffff0000ffff0000ffff0000ff202000000000000000000202
055555500550505005555550055555500005500005555550055005500550055000000000ffffffffffffffffffffffffffffffff202000000000000000000202
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
00066000000cc000000bb000000aa000000ee000000000000000000000000000766666657cccccc17bbbbbb37aaaaaa97eeeeee8000000000000000000000000
0060060000c00c0000b00b0000a00a0000e00e00000000000000000000000000766766657cc7ccc17bb7bbb37aa7aaa97ee7eee8000000000000000000000000
0060060000c00c0000b00b0000a00a0000e00e00000000000000000000000000766656657ccc3cc17bbb3bb37aaa9aa97eee8ee8000000000000000000000000
00066000000cc000000bb000000aa000000ee000000000000000000000000000766666657cccccc17bbbbbb37aaaaaa97eeeeee8000000000000000000000000
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
1d00000000000000000000000000001f1d38383838383838383838383838001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f1d38060605060506050605060638001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f1d38383801380238033804383838001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f1d00003801380238033804380000001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f1d00003801380238033804380000001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f1d00003801380238033804380000001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
1d00000000000000000000000000001f1d00003839383a383b383c380000001f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
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
