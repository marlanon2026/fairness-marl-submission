(define (problem robotouille)
(:domain robotouille)
(:objects
    hospital_cart_left1 - station
    patient_legs1 - station
    patient_bed_station1 - station
    hospital_cart_right1 - station
    hospital_cart1 - station
    table1 - station
    syringe1 - item
    patient1 - item
    aed1 - item
    pump1 - item
    cpr_board1 - item
    robot1 - player
    robot2 - player
    robot3 - player
)
(:init
    (ishospital_cart_left hospital_cart_left1)
    (ispatient_legs patient_legs1)
    (ispatient_bed_station patient_bed_station1)
    (ishospital_cart_right hospital_cart_right1)
    (ishospital_cart hospital_cart1)
    (istable table1)
    (issyringe syringe1)
    (issyringeusable syringe1)
    (ispatient patient1)
    (ispatient patient1)
    (ischestcompressable patient1)
    (isaed aed1)
    (isusableforaed aed1)
    (ispump pump1)
    (ispumpusable pump1)
    (iscpr_board cpr_board1)
    (iscpr_board cpr_board1)
    (isrobot robot1)
    (isrobot robot2)
    (isrobot robot3)
    (at syringe1 hospital_cart_left1)
    (loc robot1 hospital_cart_left1)
    (empty patient_legs1)
    (vacant patient_legs1)
    (at patient1 patient_bed_station1)
    (vacant patient_bed_station1)
    (at aed1 hospital_cart_right1)
    (at pump1 hospital_cart_right1)
    (loc robot2 hospital_cart_right1)
    (at cpr_board1 hospital_cart1)
    (vacant hospital_cart1)
    (empty table1)
    (loc robot3 table1)
    (nothing robot1)
    (nothing robot2)
    (nothing robot3)
    (selected robot1)
    (on syringe1 hospital_cart_left1)
    (clear syringe1)
    (on patient1 patient_bed_station1)
    (clear patient1)
    (on aed1 hospital_cart_right1)
    (atop pump1 aed1)
    (clear pump1)
    (on cpr_board1 hospital_cart1)
    (clear cpr_board1)
    (canmoveitem robot1)    (canmove robot1)    (cancompresschest robot1)    (cangiverescuebreaths robot1)    (cangiveshock robot1)    (cangivemedicine robot1)    (canmoveitem robot2)    (canmove robot2)    (cancompresschest robot2)    (cangiverescuebreaths robot2)    (cangiveshock robot2)    (cangivemedicine robot2)    (canmoveitem robot3)    (canmove robot3)    (cancompresschest robot3)    (cangiverescuebreaths robot3)    (cangiveshock robot3)    (cangivemedicine robot3))
(:goal
   (or
       (and
           (istreated patient1)
       )
   )
)
