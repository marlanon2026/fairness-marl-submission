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
        (isfryer ?s - station)
        (ispatient ?s - item)
        (isaed_station ?s - station)
        (iscpr_station ?s - station)
        (isventilation_station ?s - station)
        (ispatient_bed_station ?s - station)
        (ishospital_cart ?s - station)
        (ishospital_cart_left ?s - station)
        (ishospital_cart_right ?s - station)
        (ispatient_legs ?s - station)      
        (isrobot ?p - player)
        (isnurse ?p - player)
        (istopbun ?i - item)
        (isbottombun ?i - item)
        (isbread ?i - item)
        (islettuce ?i - item)
        (isonion ?i - item)
        (istomato ?i - item)
        (iscuttable ?i - item)
        (iscut ?i - item)
        (ispatty ?i - item)
        (ischicken ?i - item)
        (iscookable ?i - item)
        (iscooked ?i - item)
        (ischeese ?i - item)
        (isfryable ?i - item)
        (isfryableifcut ?i - item)
        (isfried ?i - item)
        (ispotato ?i - item)
        (isusableforcpr ?i - item)
        (isusedforcpr ?i - item)
        (isusedforaed ?i - item)
        (isventilator ?i - item)
        (isusableforventilation ?i - item)
        (isusedforventilation ?i - item)
        (isusableforaed ?i - item)
        (ismedicine ?i - item)
        (isaed ?i - item)
        (iscpr_kit ?i - item)
        (iscpr_stool ?i - item)
        (iscpr_stoolusuable ?i - item)
        (iscpr_stoolused ?i - item)
        (iscpr_board ?i - item)
        (iscpr_boardused ?i - item)
        (iscpr_boardusuable ?i - item)
        (ischestcompressable ?i - item)
        (ischestcompressed ?i - item)
        (isrbifchestcompressed ?i - item) 
        (iseligibletoreceiverescuebreaths ?i - item)
        (isrescuebreathed ?i - item)
        (istreated ?i - item)
        (issyringe ?i - item)
        (issyringeusable ?i - item)
        (issyringeused ?i - item)
        (ispump ?i - item)
        (ispumpusable ?i - item)
        (ispumpused ?i - item)
        (isshocked ?i - item)
        (istired ?i - player)

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
        (cprboard-properly-placed ?i - item ?s - station)
        (cancook ?p - player)
        (cancut ?p - player)
        (canmoveitem ?p - player)
        (canmove ?p - player)
        (canfry ?p - player)
        (canfrycut ?p - player)
        (cangivemedicine ?p - player)
        (cancompresschest ?p - player)
        (cangiverescuebreaths ?p - player)
        (cangiveshock ?p - player)
    )

    ; ACTIONS
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
    
    ; Make the player stack an item on top of another item at a station
    (:action stack
        :parameters (?p - player ?i1 - item ?i2 - item ?s - station)
        :precondition (and
            (has ?p ?i1)
            (clear ?i2)
            (loc ?p ?s)
            (at ?i2 ?s)
            (selected ?p)
            (canmoveitem ?p)
        )
        :effect (and
            (nothing ?p)
            (at ?i1 ?s)
            (atop ?i1 ?i2)
            (clear ?i1)
            (not (clear ?i2))
            (not (has ?p ?i1))
        )
    )

    ; Make the player unstack an item from another item at a station
    (:action unstack
        :parameters (?p - player ?i1 - item ?i2 - item ?s - station)
        :precondition (and
            (nothing ?p)
            (clear ?i1)
            (atop ?i1 ?i2)
            (loc ?p ?s)
            (at ?i1 ?s)
            (at ?i2 ?s)
            (selected ?p)
            (canmoveitem ?p)
            (not (cprboard-properly-placed ?i2 ?s))  ; Prevent unstacking properly placed items
        )
        :effect (and
            (has ?p ?i1)
            (clear ?i2)
            (not (nothing ?p))
            (not (clear ?i1))
            (not (atop ?i1 ?i2))
            (not (at ?i1 ?s))
        )
    )

    ; Make the player cook a cookable item on a stove
    (:action cook
        :parameters (?p - player ?i - item ?s - station)
        :precondition (and
            (isstove ?s)
            (iscookable ?i)
            (on ?i ?s)
            (loc ?p ?s)
            (clear ?i)
            (selected ?p)
            (cancook ?p)
        )
        :effect (and
            (iscooked ?i)
        )
    )

    ; Make the player fry a fryable item in a fryer
    (:action fry
        :parameters (?p - player ?i - item ?s - station)
        :precondition (and
            (isfryer ?s)
            (isfryable ?i)
            (on ?i ?s)
            (loc ?p ?s)
            (clear ?i)
            (selected ?p)
            (canfry ?p)
        )
        :effect (and
            (isfried ?i)
        )
    )

    ; Make the player fry an item that is only fryable if cut, in a fryer
    (:action fry_cut_item
        :parameters (?p - player ?i - item ?s - station)
        :precondition (and
            (isfryer ?s)
            (isfryableifcut ?i)
            (iscut ?i)
            (on ?i ?s)
            (loc ?p ?s)
            (clear ?i)
            (selected ?p)
            (canfrycut ?p)
        )
        :effect (and
            (isfried ?i)
        )
    )
    
    ; Make the player cut a cuttable item on a cutting board
    (:action cut
        :parameters (?p - player ?i - item ?s - station)
        :precondition (and
            (isboard ?s)
            (iscuttable ?i)
            (on ?i ?s)
            (loc ?p ?s)
            (clear ?i)
            (selected ?p)
            (cancut ?p)
        )
        :effect (and
            (iscut ?i)
        )
    )

    ; Make the player stack an item under another item at a station
    (:action stackunder
        :parameters (?p - player ?i1 - item ?i2 - item ?s - station)
        :precondition (and
            (has ?p ?i1)
            (loc ?p ?s)
            (on ?i2 ?s)
            (selected ?p)
            (canmoveitem ?p)
            (iscpr_board ?i1)  ; Ensure i1 is a CPR board
            (ispatient ?i2)    ; Ensure i2 is a patient
        )
        :effect (and
            (nothing ?p)
            (at ?i1 ?s)
            (on ?i1 ?s)
            (atop ?i2 ?i1)
            (not (on ?i2 ?s))
            (not (has ?p ?i1))
            (cprboard-properly-placed ?i1 ?s)
            )
        )
    )

    ;Make the player compress patient's chest
    (:action compresschest
        :parameters (?p - player ?i - item ?i2 - item ?s - station)
        :precondition (and
            (ispatient_bed_station ?s)
            (ispatient ?i) 
            (ischestcompressable ?i)
            (iscpr_board ?i2)
            (on ?i2 ?s)
            (atop ?i ?i2) ; Ensure item ?i2 is atop patient ?i
            (loc ?p ?s)
            (clear ?i)
            (selected ?p)
            (cancompresschest ?p)
            (not (istired ?p))
            (not (ischestcompressed ?i))
            (cprboard-properly-placed ?i2 ?s)
        )
        :effect (and
            (ischestcompressed ?i)
        )
    )

    ;Make the player give patient rescue breaths
    (:action giverescuebreaths
        :parameters (?p - player ?i - item ?i2 - item ?i3 - item ?s - station)
        :precondition (and
            (ispatient_bed_station ?s)
            (ispatient ?i) 
            (iscpr_board ?i2) ; Ensure ?i2 is the cpr_board - New line    
            (ispump ?i3)
            (ispumpusable ?i3)
            (at ?i3 ?s)
            (ischestcompressed ?i)  ; Typically, rescue breaths follow chest compressions
            (on ?i2 ?s)
            (atop ?i ?i2) ; Ensure item ?i2 cpr_board is atop patient ?i - Modified line 
            (atop ?i3 ?i)       
            (loc ?p ?s)
            (selected ?p)
            (clear ?i3) ; pump is at the top
            (cangiverescuebreaths ?p)
            (not (isrescuebreathed ?i))
            (cprboard-properly-placed ?i2 ?s)
        )
        :effect (and
            (isrescuebreathed ?i)
        )
    )

    ; Deliver shock
    (:action giveshock
        :parameters (?p - player ?i1 - item ?i2 - item ?i3 - item ?i4 - item ?s - station)
        :precondition (and
            (ispatient_bed_station ?s)
            (ispatient ?i1)
            (isaed ?i2)
            (ispump ?i3)
            (iscpr_board ?i4)
            (isrescuebreathed ?i1)
            (isusableforaed ?i2)
            (ispumpusable ?i3)
            (on ?i4 ?s)
            (at ?i4 ?s)
            (at ?i3 ?s)
            (atop ?i3 ?i1)
            (atop ?i2 ?i3)
            (loc ?p ?s)
            (selected ?p)
            (not (isshocked ?i1))
            (cangiveshock ?p)
        )
        :effect (and
            (isshocked ?i1)    
        )
    )

    ; Insert Syringe into the patient,; Make the nurse player place a medicine item on top the patient station
    (:action givemedicine
        :parameters (?p - player ?i1 - item ?i2 - item ?i3 - item ?i4 - item ?i5 - item ?s - station)
        :precondition (and
            (ispatient_bed_station ?s)
            (ispatient ?i1)
            (isaed ?i2)
            (ispump ?i3)
            (iscpr_board ?i4)
            (issyringe ?i5)
            (isshocked ?i1)
            (isusableforaed ?i2)
            (ispumpusable ?i3)
            (issyringeusable ?i5)
            (on ?i4 ?s)
            (at ?i4 ?s)
            (at ?i3 ?s)
            (at ?i5 ?s)
            (atop ?i3 ?i1) ;pump is on the patient 
            (atop ?i2 ?i3) ;AED is on the pump
            (atop ?i5 ?i2) ;syringe is on the AED
            (loc ?p ?s)
            (selected ?p)
            (not (istreated ?i1))
            (cangivemedicine ?p)
        )
        :effect (and
            (istreated ?i1)
        )
    )


    ; SIMPLIFIED VERSION - Remove complex dependencies

    ; Direct chest compression (no CPR board required)
    (:action compresschest_simple
        :parameters (?p - player ?i - item ?s - station)
        :precondition (and
            (ispatient_bed_station ?s)
            (ispatient ?i) 
            (ischestcompressable ?i)
            (loc ?p ?s)
            (selected ?p)
            (cancompresschest ?p)
            (not (istired ?p))
            (not (ischestcompressed ?i))
        )
        :effect (and
            (ischestcompressed ?i)
        )
    )

    ; Direct rescue breaths (no pump required)
    (:action giverescuebreaths_simple
        :parameters (?p - player ?i - item ?s - station)
        :precondition (and
            (ispatient_bed_station ?s)
            (ispatient ?i)
            (ischestcompressed ?i)  ; Still requires compressions first
            (loc ?p ?s)
            (selected ?p)
            (cangiverescuebreaths ?p)
            (not (isrescuebreathed ?i))
        )
        :effect (and
            (isrescuebreathed ?i)
        )
    )

    ; Simple shock - requires rescue breaths but no equipment
    (:action giveshock_simple
        :parameters (?p - player ?i - item ?s - station)
        :precondition (and
            (ispatient_bed_station ?s)
            (ispatient ?i)
            (isrescuebreathed ?i)  ; Must have rescue breaths completed
            (loc ?p ?s)
            (selected ?p)
            (not (isshocked ?i))
            (cangiveshock ?p)
        )
        :effect (and
            (isshocked ?i)    
        )
    )

    ; Simple medicine - requires shock but no equipment
    (:action givemedicine_simple
        :parameters (?p - player ?i - item ?s - station)
        :precondition (and
            (ispatient_bed_station ?s)
            (ispatient ?i)
            (isshocked ?i)  ; Must have shock completed
            (loc ?p ?s)
            (selected ?p)
            (not (istreated ?i))
            (cangivemedicine ?p)
        )
        :effect (and
            (istreated ?i)
        )
    )

    (:action compresschest_preplaced
        :parameters (?p - player ?i - item ?i2 - item ?s - station)
        :precondition (and
            (ispatient_bed_station ?s)
            (ispatient ?i) 
            (ischestcompressable ?i)
            (iscpr_board ?i2)
            (on ?i2 ?s)  ; CPR board already there
            (atop ?i ?i2) ; Patient already on board
            (loc ?p ?s)
            (selected ?p)
            (cancompresschest ?p)
            (not (istired ?p))
            (not (ischestcompressed ?i))
        )
        :effect (and
            (ischestcompressed ?i)
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
