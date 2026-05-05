(define (problem robotouille)
(:domain robotouille)
(:objects
    patient_legs1 - station
    patient_bed_station1 - station
    table1 - station
    patient1 - item
    robot1 - player
)
(:init
    (ispatient_legs patient_legs1)
    (ispatient_bed_station patient_bed_station1)
    (istable table1)
    (ispatient patient1)
    (ispatient patient1)
    (ischestcompressable patient1)
    (isrobot robot1)
    (empty patient_legs1)
    (vacant patient_legs1)
    (at patient1 patient_bed_station1)
    (vacant patient_bed_station1)
    (empty table1)
    (loc robot1 table1)
    (nothing robot1)
    (selected robot1)
    (on patient1 patient_bed_station1)
    (clear patient1)
    (canmove robot1)    (cancompresschest robot1))
(:goal
   (or
       (and
           (ischestcompressed patient1)
       )
   )
)
