
Input = {
    'L68',
    'L30',
    'R48',
    'L5',
    'R60',
    'L55',
    'L1',
    'L99',
    'R14',
    'L82',
}

local RenderSpeeds = {  1,  1,  1, 1, 2, 2, 2, 4, 4, 8, 16, 32, 64, 128, 256, 256, 256, 256, 256, 256, 256, 256 }
local RenderSkips =  {  1,  1,  1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,   1,   1,   2,   4,   8,  16,  32,  64, 128 }
local RenderDelays = { 60, 30, 15, 8, 8, 4, 2, 2, 1, 1,  1,  1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1 }

local RenderIndex = 1

Rotations = {}

for i = #Input, 1, -1 do
    local rotation = tonumber(string.sub(Input[i], 2))
    if string.sub(Input[i], 1, 1) == 'L' then
        rotation = -1 * rotation
    end
    table.insert(Rotations, rotation)
end

hooks = {
    ['HP'] = { type = 'u32', address = 0x097BA0 },
    ['HP Display'] = { type = 'u32', address = 0x13796C },
    ['Heart'] = { type = 'u32', address = 0x097BA8 },
    ['Hours'] = { type = 'u32', address = 0x097C30 },
    ['Minutes'] = { type = 'u32', address = 0x097C34 },
    ['Seconds'] = { type = 'u32', address = 0x097C38 },
    ['Frames'] = { type = 'u32', address = 0x097C3C },
}

local dial = 50
local zero_counts = 0

function update()
    if #Rotations > 0 then
        local rotation = Rotations[#Rotations]
        if rotation > 0 then
            local step = 1
            if rotation > RenderSkips[RenderIndex] then
                step = RenderSkips[RenderIndex]
            end
            dial = dial + step
            if dial > 1200 then
                dial = dial - 1200
            end
            Rotations[#Rotations] = rotation - step
        elseif rotation < 0 then
            local step = 1
            if rotation < -RenderSkips[RenderIndex] then
                step = RenderSkips[RenderIndex]
            end
            dial = dial - step
            if dial < 0 then
                dial = dial + 1200
            end
            Rotations[#Rotations] = rotation + step
        end
        if Rotations[#Rotations] == 0 then
            if dial % 100 == 0 then
                zero_counts = zero_counts + 1
            end
            table.remove(Rotations)
            for i = 1, RenderDelays[RenderIndex] do
                render()
            end
            RenderIndex = RenderIndex + 1
            if RenderIndex > #RenderSpeeds then
                RenderIndex = #RenderSpeeds
            end
            if RenderIndex > #Rotations then
                RenderIndex = #Rotations
            end
        end
    end
end

function render()
    local dial = math.floor(60 * (dial / 100))
    mainmemory.write_u32_le(hooks['HP'].address, #Rotations)
    mainmemory.write_u32_le(hooks['HP Display'].address, #Rotations)
    mainmemory.write_u32_le(hooks['Minutes'].address, dial)
    mainmemory.write_u32_le(hooks['Heart'].address, zero_counts)
    emu.frameadvance()
end

while true do
    for i = 1, RenderSpeeds[RenderIndex] do
        update()
    end
    render()
end