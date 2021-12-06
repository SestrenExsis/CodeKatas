
{
    init: function(elevators, floors) {
        let idleFloorNum = 0;
        for (const elevator of elevators) {
            elevator.on("idle", function() {
                elevator.goToFloor(idleFloorNum);
                idleFloorNum += 1;
            });
    
            elevator.on("floor_button_pressed", function(floorNum) {
                elevator.goToFloor(floorNum)
            });

            elevator.on("passing_floor", function(floorNum, direction) {});

            elevator.on("stopped_at_floor", function() {});
        }
    },
    
    update: function(dt, elevators, floors) {
        // We normally don't need to do anything here
    }
}
