(define (problem robotouille)
(:domain robotouille)
(:objects
    patient_bed_station1 - station
    hospital_cart1 - station
    cpr_board1 - item
    robot1 - player
)
(:init
    (ispatient_bed_station patient_bed_station1)
    (ishospital_cart hospital_cart1)
    (iscpr_board cpr_board1)
    (iscpr_board cpr_board1)
    (isrobot robot1)
    (empty patient_bed_station1)
    (vacant patient_bed_station1)
    (at cpr_board1 hospital_cart1)
    (vacant hospital_cart1)
    (nothing robot1)
    (selected robot1)
    (on cpr_board1 hospital_cart1)
    (clear cpr_board1)
    (canmoveitem robot1)    (canmove robot1))
(:goal
   (or
       (and
           (on cpr_board1 patient_bed_station1)
       )
   )
)
