
{
    init: function(elevators, floors) {
        for (const elevator of elevators) {
            elevator.on("idle", function() {
                elevator.goToFloor(0);
            });
    
            elevator.on("floor_button_pressed", function(floorNum) {
                elevator.goToFloor(floorNum)
            });
        }
    },
    
    update: function(dt, elevators, floors) {
        // We normally don't need to do anything here
    }
}
