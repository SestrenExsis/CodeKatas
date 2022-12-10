pico-8 cartridge // http://www.pico-8.com
version 38
__lua__
-- advent of code 2022 day 10
-- cathode-ray tube

-- notes
-->8
-- data

_defaultinput={
	"addx 15",
	"addx -11",
	"addx 6",
	"addx -3",
	"addx 5",
	"addx -1",
	"addx -8",
	"addx 13",
	"addx 4",
	"noop",
	"addx -1",
	"addx 5",
	"addx -1",
	"addx 5",
	"addx -1",
	"addx 5",
	"addx -1",
	"addx 5",
	"addx -1",
	"addx -35",
	"addx 1",
	"addx 24",
	"addx -19",
	"addx 1",
	"addx 16",
	"addx -11",
	"noop",
	"noop",
	"addx 21",
	"addx -15",
	"noop",
	"noop",
	"addx -3",
	"addx 9",
	"addx 1",
	"addx -3",
	"addx 8",
	"addx 1",
	"addx 5",
	"noop",
	"noop",
	"noop",
	"noop",
	"noop",
	"addx -36",
	"noop",
	"addx 1",
	"addx 7",
	"noop",
	"noop",
	"noop",
	"addx 2",
	"addx 6",
	"noop",
	"noop",
	"noop",
	"noop",
	"noop",
	"addx 1",
	"noop",
	"noop",
	"addx 7",
	"addx 1",
	"noop",
	"addx -13",
	"addx 13",
	"addx 7",
	"noop",
	"addx 1",
	"addx -33",
	"noop",
	"noop",
	"noop",
	"addx 2",
	"noop",
	"noop",
	"noop",
	"addx 8",
	"noop",
	"addx -1",
	"addx 2",
	"addx 1",
	"noop",
	"addx 17",
	"addx -9",
	"addx 1",
	"addx 1",
	"addx -3",
	"addx 11",
	"noop",
	"noop",
	"addx 1",
	"noop",
	"addx 1",
	"noop",
	"noop",
	"addx -13",
	"addx -19",
	"addx 1",
	"addx 3",
	"addx 26",
	"addx -30",
	"addx 12",
	"addx -1",
	"addx 3",
	"addx 1",
	"noop",
	"noop",
	"noop",
	"addx -9",
	"addx 18",
	"addx 1",
	"addx 2",
	"noop",
	"noop",
	"addx 9",
	"noop",
	"noop",
	"noop",
	"addx -1",
	"addx 2",
	"addx -37",
	"addx 1",
	"addx 3",
	"noop",
	"addx 15",
	"addx -21",
	"addx 22",
	"addx -6",
	"addx 1",
	"noop",
	"addx 2",
	"addx 1",
	"noop",
	"addx -10",
	"noop",
	"noop",
	"addx 20",
	"addx 1",
	"addx 2",
	"addx 2",
	"addx -6",
	"addx -11",
	"noop",
	"noop",
	"noop"
}

function load_input()
	if stat(4)=="" then
		_input=_defaultinput
	else
		_input=split(stat(4),"\n",false)
	end
	_data={}
	for row in all(_input) do
		add(_data,row)
	end
	_cpu=cpu:new(_data)
	_crt.bm=1
end
-->8

function _init()
	_input=_defaultinput
	_data={}
	for row in all(_input) do
		add(_data,row)
	end
	menuitem(1,
		"load pasted data",
		load_input
	)
	_cpu=cpu:new(_data)
	_crt=crt:new(17,13,40,6)
	_spd=1
end

function _update60()
	for i=1,_spd do
		_crt:step(_cpu.x)
		_cpu:step()
	end
	if btnp(⬅️) then
		_spd=max(1,_spd\2)
	elseif btnp(➡️) then
		_spd=min(240,_spd*2)
	end
end

function _draw()
	cls()
	_crt:render()
end
-->8
-- crt

crt={}

function crt:new(
	x,  -- x pos  : number
	y,  -- y pos  : number
	wd, -- width  : number
	ht  -- height : number
	)
	-- pixels on screen
	local pix={}
	local n=ht*wd
	for i=1,n do
		add(pix,0)
	end
	-- position of the beam
	local obj={
		x=x,
		y=y,
		wd=wd,
		ht=ht,
		pix=pix,
		bm=1,
	}
	return setmetatable(
		obj,{__index=self}
	)
end

function crt:step(x)
	local px=(self.bm-1)%self.wd
	self.pix[self.bm]=0
	if (
		px>=(x-1) and
		px<=(x+1)
	) then
		self.pix[self.bm]=1
	end
	self.bm+=1
	if self.bm>#self.pix then
		self.bm=1
	end
end

function crt:render()
	local lf=self.x
	local tp=self.y
	local rt=lf+self.wd
	local bt=tp+self.ht
	rect(lf-1,tp-1,rt,bt,7)
	for y=0,self.ht-1 do
		for x=0,self.wd-1 do
			local c=1
			local idx=self.wd*y+x+1
			if self.pix[idx]>0 then
				c=12
			end
			pset(lf+x,tp+y,c)
		end
	end
	local y=(self.bm-1)/self.wd
	local x=(self.bm-1)%self.wd
	pset(lf+x,tp-1,5)
	pset(lf-1,tp+y,5)
end
-->8
-- cpu

cpu={}

function cpu:new(
	input -- input : table
	)
	local mem={}
	for raw in all(input) do
		if raw=="noop" then
			add(mem,{"noop"})
		else
			local parts=split(raw," ")
			local amt=tonum(parts[2])
			add(mem,{"addx",amt})
		end
	end
	local obj={
		x=1,
		mem=mem,
		pc=1,
		delay=0,
	}
	return setmetatable(
		obj,{__index=self}
	)
end

function cpu:step()
	local instr=self.mem[self.pc]
	local op=instr[1]
	if op=="noop" then
		-- noop takes 1 clock cycle
		self.pc += 1
	elseif op=="addx" then
		-- addx takes 2 clock cycles
		if self.delay==0 then
			self.delay=1
		else
			self.delay-=1
			if self.delay<1 then
				self.x+=instr[2]
				self.pc+=1
			end
		end
	end
	if self.pc>#self.mem then
		self.pc=1
		self.x=1
	end
end
__gfx__
11111111111111111111111111111111111111110000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
11111111111111111111111111111111111111110000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
11111111111111111111111111111111111111110000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
11111111111111111111111111111111111111110000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
11111111111111111111111111111111111111110000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
11111111111111111111111111111111111111110000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
