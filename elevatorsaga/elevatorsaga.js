
{
    init: function(elevators, floors) {
        var elevator = elevators[0]; // Let's use the first elevator
        elevator.on("idle", function() {
            elevator.goToFloor(0);
        });

        elevator.on("floor_button_pressed", function(floorNum) {
            elevator.goToFloor(floorNum)
        });
    },
    
    update: function(dt, elevators, floors) {
        // We normally don't need to do anything here
    }
}
