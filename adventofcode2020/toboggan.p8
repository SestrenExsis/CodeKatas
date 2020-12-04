pico-8 cartridge // http://www.pico-8.com
version 29
__lua__
-- toboggan
-- by sestrenexsis
-- https://github.com/sestrenexsis/codekatas

-- from advent of code 2020
-- day 03

_defaultdata={
	".......#..#....#...#...#......#",
	"..##..#...##.###.#..#.....#.#..",
	"#..#.#....#......#..#.........#",
	".#..##...#........#....#..#..#.",
	"#.#.#....###...#........#.....#",
	".#...#.#.##.#.##...#.#.........",
	"####......#.......###.##.#.....",
	"..#...........#...#.#.#........",
	".#.......#....###.####..#......",
	"...##........#....##.......##..",
	".###......##.#......##....#.#.#",
	"........#.#......##...#......#.",
	"#....##.#..#...#.......#.......",
	".#..##........##.........#....#",
	".#..#..#...#....#.#......#.#...",
	"..#.#......##.#.......#....##..",
	"......##......#.#..##.#..#...#.",
	".....##.......#.#....#.#.......",
	"........#.....#.....#..###.#...",
	"#........#..#.....#...#.#.#..#.",
	".#..#.....#...#........#.....#.",
	".#.#.....#.....#...#...........",
	".....#.#..#..#...#..#..#..##..#",
	"##.#...#....#..#.##..#.....#.#.",
	"#.......####......#..#.#....#..",
	"......#.#...####.........#.#..#",
	".#.........#..#.#...#..........",
	"...#####.#....#.#..#......#.#.#",
	"##....#.###....##...##..#.....#",
	"...........####.##.#....##.##..",
	"#.#.#..........#.#..##.#.######",
	"##...#..#...........###..#....#",
	".#.#.#...##..........##.#...#..",
	"...#.#........#..##...#....#...",
	"......#..#...#..##....#.......#",
	".#..#.......#..#......##....##.",
	".......#.......#........#..##..",
	"...#...#...#.##......#.##.#....",
	".........#.........#.#.#.##....",
	"..#...................#....#..#",
	".........#..#.....#.#...#....#.",
	"#.#.#...#........#..###.#......",
	"#.#.#.####......##...#...#....#",
	"#...........##..#.#.#....#..#..",
	"........#..#.#...........##.#.#",
	".#.........#...........#..#....",
	"#............##.#..#....##...##",
	".#....##..#.#....#.......#..#..",
	"..#.#...#.#......####.......#..",
	"...#.#.......###......#.....#..",
	"#......#.......#.#...#.#..##...",
	"...#.....#...##.#.....#.#......",
	"#.#.#............#..#......#..#",
	"....#...#...##.##.##...##.#....",
	"..##........#..#........#...##.",
	".......#..#...#.........#.....#",
	"...........#.#......#...#......",
	"...##..##..##..###..#..#..#..#.",
	"#..##.......##..#....#....#.#..",
	"#.#.##.#..##.....#....#.#......",
	"....#..##......#.#..#....#....#",
	".#.#.........##...#......##.##.",
	"##...........#..#.....#.###....",
	".#.###........#...#....##..#...",
	"......##.....#.................",
	".#.##..#.#.......#......#.#.#..",
	".#...#....#.##..........##.##..",
	"#...##......####.#....#....#...",
	".#...#.##.#.#.....#...#........",
	".#................#.##.#.###...",
	"...#.#..#.#.....##.....##....#.",
	"..##.#..#..##.....#....#...#.##",
	"........###.##..#..###.....#..#",
	"..##.....#.......#.#...##......",
	"#.#..###...##.###.##.#..#...#..",
	"#..#..#.#...#....#...##.....#.#",
	"#..................#........#..",
	"#.....#.......#.##....##....#..",
	"...#.............#.....#...#...",
	"...#...#.##..##.....#........#.",
	".......#........##....###..##..",
	".#....#....#.#..#......#....#.#",
	"..........#..#.#.....##...#.##.",
	".#...##.#...........#.#.......#",
	"..#.##.....#.###.#.............",
	"..#....###..........#.#.#......",
	"#.....#.####..#.#......#..#.#.#",
	"...#........#..#...............",
	".###.#.##.....#.#...........#..",
	"..#....#..#....#..##....#......",
	"......#..#.....#.#.##.......#.#",
	"###..#...#.#..#....#..##.###..#",
	".#....##.###........##...##.#.#",
	"........##..##.#....##..#....#.",
	"...#..#....#.#....#...#...##...",
	"#.....#......#.##........#....#",
	"....#....###.##...#.#.##....#..",
	"......#.##..#.#..........#...#.",
	"...........#...#....##...#....#",
	"......#.#.........#....#.#.#...",
	".###..........#.###.##....#...#",
	"...##.......#......#....#....#.",
	"#..#...#.#..####...#......#..#.",
	"....##..#.#.........#..........",
	".##.###.##....##.####....#...#.",
	"..##.......#........#...#..#...",
	"....#####..........###....#....",
	".#.#..#.#.#....#..#............",
	"........#.....#....#.......##..",
	"...........##....##..##.....##.",
	"..###........#.#.#..#....##...#",
	".....#...........##......#..#..",
	"...##........#.##.#......##..#.",
	"##..#....#............##..#..#.",
	".#.....#...##.##..............#",
	"#..##........#...#...#......##.",
	"......##.....#.......####.##..#",
	"...#.#....#...#..#.............",
	"..#...#..##.###..#..#.......##.",
	"##....###.......#...#..#.......",
	"#..#.....###.....#.#.........#.",
	"#.#....#.............#...#.....",
	"..#.#.##..........#.....##.#...",
	".....##......#..#..#.....#..#..",
	"##.#..#..#.##......###....#..#.",
	"...#............##...#..##.....",
	".#..#....#.........#......#.##.",
	".##.##...#..............#..#.##",
	"...#....#...###...#...#....#..#",
	"..#...#..####..#....#.#...##..#",
	"..............##.##.......##...",
	"..##.#..##...........#.#.#...#.",
	"..................##.####.###..",
	".#...........#.......#......#..",
	".#.#.#...#....#.........##...##",
	"....#..........#.#....#.#.....#",
	"..........#.#..........#.#.....",
	"...........#.....#.#......#....",
	"........#..#.#.#.#.............",
	"...###...##...##..####.##......",
	".#..#......###.....#...#.....#.",
	".........##............#.#.....",
	"#.#..#.#.#....###.#.#..#..#..##",
	"..........#...#.##.#..#..#....#",
	"#..#.......##....#..##........#",
	"##.#...#....##.............#...",
	"....#........#......##..#..#.##",
	".................#.#.#.#.#.....",
	"...........#.#.....#.......#...",
	"#.......#.......#............#.",
	"....#...........#.#.##.....#..#",
	"#...#.....#....#..##...#.......",
	"..#.....#.....#.##.##....#.....",
	".#.#..#...#..#..##.....##..#...",
	".#.#....#.........####.........",
	"#...#..####.....#...#..##......",
	"..#...##.#.....#...#.....##....",
	".#...#.....#.#.#......#.......#",
	"..#.....##.#..#.#...##.........",
	"##.#...#..#....#....#.##.##...#",
	".#..#....#..##.#.......#..#....",
	"...##.#......#...###.......#...",
	"...#..#.........##.####........",
	"#.#..#..##...........#..#......",
	".#...#.#......#.#..........#...",
	"...###...#.......#.....#.#...##",
	"..#....#.#.##..........##...#..",
	".....###.........#.....#..##..#",
	".......##.....#.#.....#.#..##..",
	".#.#.###..##.......##...#......",
	"......#.....#................##",
	".#......##..##.#.#...#...#...##",
	".#...#......#.......#.#........",
	".#..........###...#..#...#.....",
	".........##.....#.#..#..#.#...#",
	"#...#...#.........#..#..#....#.",
	"###.......#.#.....#....##......",
	".#..#......#..#...........#..#.",
	"..##....##..##...#......#......",
	".#........#....#...#....#.....#",
	".#.......#...#...#..##.#.#..#..",
	"#...#........#.##.....#.....#..",
	"#..##.....#..........#...#...##",
	"............#...............#..",
	".#.##...#.....#.#..#..#..#.....",
	".#.#.#...#........#....#...##..",
	"##......#.....#.###.#...#.#..#.",
	".........##..#..#.#...#...#...#",
	"#...#.#....#..#..#.....#.......",
	".......#.###...#.............#.",
	"..#.....#.#.#..###.#....#.....#",
	"....#...#.#....#.#..........#..",
	"..#......#.###.#.#..#.....#...#",
	"#............#..##...##......#.",
	"#...........#..#....#.###..###.",
	".#.##.#.#.......#.............#",
	"..............#................",
	"..#.#.....#.....#...#......#...",
	".#.#.#..#..#.#...........##....",
	".....##.#......#..#.##....#....",
	".......##..#.#.#..#............",
	"..#.....#.....#.###..#.....#.#.",
	"......##.....#..##.#...#.....#.",
	"...#...#....#..#..#........#...",
	"..#.##..#....#.........#.#..#..",
	"#....#.....###.....#......#....",
	"##.....#..#..##.........#.##.##",
	".#.#....#.#..........#.........",
	".##.#...#..#.......#.##...#....",
	"...#...#.....#....#...#.#..#...",
	".....#....#.....#.....#.#......",
	"...........#.#.......#.......#.",
	".........##.###.##........#....",
	"#..##.....#...#.#..............",
	".#...#....##........#.#..#....#",
	"..#...#........#...#..#.##.#..#",
	"........#...#.....##.#.#....#.#",
	"#..#.......###.#....#.#.#......",
	".......#...##....#...#..##..#..",
	".....##........#.#.#..#....##..",
	".#....#..#.#...........#......#",
	"...##....#.##.....##.......#...",
	".##..#..#....#.#....#..#....##.",
	"..#....#.....###.......#..##..#",
	"....#.......#....##..#....#..##",
	"....#......##..#....#.#...#.#..",
	".##.#......##..................",
	"##.#....#........#..#..#...##.#",
	".......#..#.#...##.....#.#.....",
	"..##.#...........#.#.#..#.#.#..",
	".....#....#......#..#.......#..",
	"#.#...#.####..##.......#..##...",
	"...#....#.....#.##.#..#.##..#..",
	".#.......#......##........##.#.",
	".......#.#...#..#...#..##.#....",
	".#....#........#.#.....##..#..#",
	"#..#.....#..#.............#...#",
	"#...#....#..#...###..#...#.#...",
	".#..#.....#..........#..##.####",
	"#.#.#.#.##.#.#.....##.#........",
	"...#....##....#...#..##.......#",
	"..##.##.#.#........#..........#",
	"..###........###..#..........#.",
	"...#......#..##.#........#..#..",
	"#.#.#..#........#..#..........#",
	"...#........#..##.#...#.###....",
	"##......#.####.#....#......#...",
	".#..#......#................#..",
	"#.#........#.#.....##.....##...",
	"#...............#..#.......#.#.",
	".##..#...........##..#..#.#....",
	"#......#.#.......#.#.#.##..#.##",
	".....##.#..###.............##..",
	"....##.........#..#...#........",
	".....#.....#.#.#..#.#..........",
	"#.........#....##.#.##.....#..#",
	".#.........#......#.#.##.#.#...",
	"##.........#.....#..#.#..#.##.#",
	"....#......##...#.....#..#..###",
	"..#..............#...#..####...",
	"#....#...##.#.......#...#..#...",
	"#.......###.#.#.......#.......#",
	"...##....#.#...........#...###.",
	"...........#..#.#.....#..##..#.",
	"..#.........#..###..#.....#...#",
	"..#.#.....#.#.#...#.#.#......#.",
	"........#.....#.#......##....##",
	"##.#.#...#.#........#.....#...#",
	"........#....#...............#.",
	"##.###......####...#####..#....",
	"...##...#..#....#........#...#.",
	"...###.#..................##.#.",
	"##.#.......###.......#...#.#...",
	"....#..#.#...#...#....#.#.#..##",
	"....#...........#..#...........",
	"#..#.#..#...#...#..#...........",
	"...#...#.#....#..#....#........",
	"#....#.......#.##........#..#..",
	".....#...#..#................#.",
	"#......#.......#..........##..#",
	".#....#.#......#.#...#....##..#",
	"...#.##...#......#.#...##...##.",
	"..#...#..##...#...#....#.......",
	".....#....#.#.#..........#.#...",
	"...#...#..#....#..#.#..........",
	"......#.#..........##.......#..",
	".#...##.#.#...#..##..#...#.....",
	"..#..#.........#........#.#.#..",
	"#.#..##..#.....##......#.....#.",
	"#..#.....#.#....#...#.#....#.#.",
	"......#........##.#..#...#.....",
	"...#.##.#.#......#.#..##...#..#",
	"....#..###..#..#.....###....##.",
	".....#...#.#.....#..........#.#",
	".#...##..##.....#..#...#.#.#...",
	".##.#......##...##..#...#.....#",
	".#.##....#...#.##.#.#...#.#...#",
	"....#.#...#....###.#.....#.....",
	"#.....####................#..#.",
	"....#.....#...#.#.......##.#...",
	".#...##.#...#..#...........#.#.",
	"..#####..#.#...#...##........#.",
	"...#...##........#...#.#....###",
	"........#.#.#..#.....#.......#.",
	"...#...#..##............##.....",
	"#.#..###....###.#...#.#...##.##",
	"..#.##...#......#..#.........##",
	".##..#..#.....#..#.........#.#.",
	".#..#.#....#.##...#..#.##....##",
	"..#...#.#...##.#.#...#...#....#",
	"#..........#.......##..##....#.",
	"#...###.#......#....#.........#",
	"#.....#...##.......##....##....",
	".##.#..#.##......#.##....#..#..",
	"............#.#....##.#..#....#",
	".#.........##.##...#....#.....#",
	"##....##..#..#....##...#.....##",
	"...#.....#...........#.....##..",
	"......#...#.........#.......#..",
	"............#...##.#.....#.#.#.",
	".#........##..........#.....#.#",
	".###.........#.....#.##...#....",
	".##..#...##...#..#..#.##.......",
	}
-->8
-- main

function _init()
	menuitem(1,
		"load input",
		load_input
		)
	_data=_defaultdata
	_x=0
	_y=0
	_spd=0.5
	_dx=7
	_dy=1
	_shk=0
	_cx=0 -- camera shake x
	_cy=0 -- camera shake y
	-- generate a repeating
	-- quadrant of snowfields
	_qsz=32     -- quad size
	_siz=2*_qsz -- map size
	_spc=32 -- space between trees
	-- randomly generate a quadrant
	for r=0,_qsz-1 do
		for c=0,_qsz-1 do
			local idx=3+flr(rnd(5))
			idx=max(idx,3+flr(rnd(5)))
			mset(c,r,idx)
		end
	end
	-- copy left quadrant to right
	for r=0,_qsz-1 do
		for c=0,_qsz-1 do
			local idx=mget(c,r)
			mset(_qsz+c,r,idx)
		end
	end
	-- copy top half to bottom
	for r=0,_qsz-1 do
		for c=0,_siz-1do
			local idx=mget(c%_qsz,r)
			mset(c,_qsz+r,idx)
		end
	end
end

function _update60()
	local vx=_spd*_dx
	local vy=_spd*_dy
	if btn(⬅️) or btn(⬆️) then
		_x-=vx
		_y-=vy
	elseif btn(➡️) or btn(⬇️) then
		_x+=vx
		_y+=vy
	end
	if btnp(❎) then
		_shk=min(1,_shk+1)
	elseif btnp(🅾️) then
		_shk=max(0,_shk-1)
	end
	if rnd()<0.5 then
		_cx=flr(rnd(2*_shk+1))-_shk
	else
		_cy=flr(rnd(2*_shk+1))-_shk
	end
end

function _draw()
	cls(1)
	-- draw snow
	local m=8*_siz/2
	local x=_x%m
	local y=_y%m
	camera(x+_cx,y+_cy)
	map(0,0,0,0,128,128)
	-- draw trees
	camera(_x-64+_cx,_y-64+_cy)
	local rows=#_data
	local cols=#_data[1]
	for r=0,31 do
		for c=0,31 do
			local tr=flr(_y/_spc)-16
			local tc=flr(_x/_spc)-16
			local col=1+(tc+c)%cols
			if 1<=tr+r+1
			and tr+r+1<=#_data then
				local row=_data[tr+r+1]
				local cel=sub(row,col,col)
				local sy=_spc*(r+tr)
				local sx=_spc*(c+tc)
				if cel=="#" then
					spr(2,sx-4,sy-4)
					--rect(sx-1,sy-1,sx,sy,8)
				elseif cel=="x" then
					spr(18,sx-4,sy-4)
				end
			end
			--[[_spc*_y
			if r==_y and c==col then
				spr(16,_spc*c,_spc*r)
			end
			--]]
		end
	end
	-- draw toboggan
	camera(x-64+_cx,y-64+_cy)
	local idx=1
	if flr(4*t()%2)<1 then
		idx+=16
	end
	spr(idx,x-4,y-4)
	--rect(x-1,y-1,x,y,8)
	camera()
	print(_dx,1,1,1)
	print(_dy,1,7,1)
end

function load_input()
	local input=stat(4)
	if input~="" then
		_data=split(input,"\n",false)
	end
end
__gfx__
00000000088880000003300077777777777777777777777777777777777777770000000000000000000000000000000000000000000000000000000000000000
00000000608888000033b30077777777777777777777777777777777777777770000000000000000000000000000000000000000000000000000000000000000
007007000de1d10000333b007777777777cc77777777777777777777777777770000000000000000000000000000000000000000000000000000000000000000
0007700000dddd000333b3307cc777777c77c7777777777777777c77777777770000000000000000000000000000000000000000000000000000000000000000
000770000033300003b33b30c77c7777777777777777777777777777777777770000000000000000000000000000000000000000000000000000000000000000
007007000d333d403333b3b377777777777777777777cc7777777777777777770000000000000000000000000000000000000000000000000000000000000000
0000000000303004000440007777cc7777777777777c77c777777777777777770000000000000000000000000000000000000000000000000000000000000000
000000004444444000044000777c77c7777777777777777777777777777777770000000000000000000000000000000000000000000000000000000000000000
88888888608880000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
80000008088888000000000000030400003000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
800000080de1d10000000000000040000003b0000000300000000000000000000000000000000000000000000000000000000000000000000000000000000000
8000000800dddd000000000000040000000330000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
80000008003330000000000000203000000003000300040000000000000000000000000000000000000000000000000000000000000000000000000000000000
800000080d333d400000000000000000000000400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
80000008003030040002200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
88888888444444400004400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
