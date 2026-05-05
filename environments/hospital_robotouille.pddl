;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;           Robotouille!!!          ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain robotouille)
    (:requirements :strips :typing :disjunctive-preconditions)
    (:types station player item)
    (:predicates
        ; Identity Predicates
        (istable ?s - station)
        (isstove ?s - station)
        (isboard ?s - station)
        (ispatient ?s - station)
        (isaed_station ?s - station)  ; Added predicate
        (iscpr_station ?s - station)  ; Added predicate
        (isventilation_station ?s - station)  ; Added predicate
        (istreatable ?i - station)
        (istreated ?i - station)
        (isdefibrillated ?i - station)
        (isresuscitated ?i - station) 
        (isventilated ?i - station)   
        (isrobot ?p - player)
        (isnurse ?p - player)
        (isusableforcpr ?i - item)
        (isventilator ?i - item)
        (isusableforventilation ?i - item)
        (isusableforaed ?i - item)
        (ismedicine ?i - item)
        (isingestable ?i - item)
        (isaed ?i - item)
        (iscpr_kit ?i - item)
        
        
        
        ; State Predicates
        (loc ?p - player ?s - station)
        (at ?i - item ?s - station)
        (nothing ?p - player)
        (empty ?s - station)
        (on ?i - item ?s - station)
        (vacant ?s - station)
        (clear ?i - item)
        (atop ?i1 - item ?i2 - item)
        (has ?p - player ?i - item)
        (selected ?p - player)
        (canmoveitem ?p - player)
        (canmove ?p - player)
        (cangivemedicine ?p - player)
        (cangiveAED ?p - player)
        (cangiveCPR ?p - player)
        (cangivevent ?p - player)
    )

    ; ACTIONS

    ; Make the nurse player place a medicine item on top the patient station
    (:action givemedicine
        :parameters (?p - player ?i - item ?s - station)
        :precondition (and
            (ispatient ?s)
            (isingestable ?i)
            (on ?i ?s)
            (loc ?p ?s)
            (clear ?i)
        )
        :effect (and
            (istreated ?s)
        )
    ) 
     ; Make the nurse player place a CPR to the patient
    (:action giveCPR
        :parameters (?p - player ?i - item ?s - station)
        :precondition (and
            (ispatient ?s)
            (isusableforcpr ?i)
            (on ?i ?s)
            (loc ?p ?s)
            (clear ?i)
        )
        :effect (and
            (isresuscitated ?s)
        )
    ) 

    ; Make the nurse player give AED to the patient
    (:action giveAED
        :parameters (?p - player ?i - item ?s - station)
        :precondition (and
            (ispatient ?s)
            (isusableforaed ?i)
            (on ?i ?s)
            (loc ?p ?s)
            (clear ?i)
        )
        :effect (and
            (isdefibrillated ?s)
        )
    )   

    ; Make the nurse player give ventilation to the patient
    (:action giveVentilation
        :parameters (?p - player ?i - item ?s - station)
        :precondition (and
            (ispatient ?s)
            (isventilator ?i)
            (isusableforventilation ?i)
            (on ?i ?s)
            (loc ?p ?s)
            (clear ?i)
        )
        :effect (and
            (isventilated ?s)
        )
    )  

    ; Move the player from station 1 to station 2
    (:action move
        :parameters (?p - player ?s1 - station ?s2 - station)
        :precondition (and
            (loc ?p ?s1)
            (selected ?p)
            (canmove ?p)
        )
        :effect (and
            (loc ?p ?s2)
            (vacant ?s1)
            (not (loc ?p ?s1))
            (not (vacant ?s2))
        )
    )

    ; Make the player pick up an item from a station
    (:action pick-up
        :parameters (?p - player ?i - item ?s - station)
        :precondition (and 
            (nothing ?p)
            (on ?i ?s)
            (loc ?p ?s)
            (clear ?i)
            (selected ?p)
            (canmoveitem ?p)
        )
        :effect (and 
            (has ?p ?i)
            (empty ?s)
            (not (nothing ?p))
            (not (at ?i ?s))
            (not (clear ?i))
            (not (on ?i ?s))
        )
    )

    ; Make the player place an item on a station
    (:action place
        :parameters (?p - player ?i - item ?s - station)
        :precondition (and 
            (has ?p ?i)
            (loc ?p ?s)
            (empty ?s)
            (selected ?p)
            (canmoveitem ?p)
        )
        :effect (and 
            (nothing ?p)
            (at ?i ?s)
            (clear ?i)
            (on ?i ?s)
            (not (has ?p ?i))
            (not (empty ?s))
        )
    )

    ; Change player selection
    (:action select
        :parameters (?p1 - player ?p2 - player)
        :precondition (and 
            (selected ?p1)
            (not  (selected ?p2))
        )
        :effect (and 
            (not (selected ?p1))
            (selected ?p2)
        )
    )
)