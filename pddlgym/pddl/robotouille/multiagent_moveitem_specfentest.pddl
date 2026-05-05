(define (problem robotouille)
(:domain robotouille)
(:objects
    hospital_cart_left1 - station
    patient_bed_station1 - station
    hospital_cart_right1 - station
    hospital_cart1 - station
    table1 - station
    cpr_board1 - item
    pump1 - item
    robot1 - player
    robot2 - player
    robot3 - player
)
(:init
    (ishospital_cart_left hospital_cart_left1)
    (ispatient_bed_station patient_bed_station1)
    (ishospital_cart_right hospital_cart_right1)
    (ishospital_cart hospital_cart1)
    (istable table1)
    (iscpr_board cpr_board1)
    (iscpr_board cpr_board1)
    (ispump pump1)
    (ispumpusable pump1)
    (isrobot robot1)
    (isrobot robot2)
    (isrobot robot3)
    (empty hospital_cart_left1)
    (loc robot1 hospital_cart_left1)
    (empty patient_bed_station1)
    (vacant patient_bed_station1)
    (empty hospital_cart_right1)
    (loc robot2 hospital_cart_right1)
    (at cpr_board1 hospital_cart1)
    (vacant hospital_cart1)
    (at pump1 table1)
    (loc robot3 table1)
    (nothing robot1)
    (nothing robot2)
    (nothing robot3)
    (selected robot1)
    (on cpr_board1 hospital_cart1)
    (clear cpr_board1)
    (on pump1 table1)
    (clear pump1)
    (canmoveitem robot1)    (canmove robot1)    (canmoveitem robot2)    (canmove robot2)    (canmoveitem robot3)    (canmove robot3))
(:goal
   (or
       (and
           (atop cpr_board1 pump1)
       )
   )
)
