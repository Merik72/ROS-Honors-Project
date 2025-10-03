using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using RosMessageTypes.Geometry;
using System.IO;
using UnityEngine.UIElements;
using Unity.VisualScripting;
using System;
// Added for temporary implementation of shutdown command
// using UnityColor = RosMessageTypes.UnityRoboticsDemo.UnityColorMsg;

// Current theory of  issue with the robot - Time is too short, and it doesn't get rewarded enough for moving directly away from the origin.
// Observation: Dingo will drive backwards towards the wall, slightly turning over and over to avoid it, inching towards it,
// getting as close as it can before time runs out

// Proposed Solution: On top of Delta displacement, also reward total displacement again in some way.


[DefaultExecutionOrder(1)]
public class thanMove : MonoBehaviour
{
    private enum Contact
{
    Wall, NoWall
}
    // public UpdatedWheelSubscriber dingo;
    // ROSConnection publishTermination;
    [Header("Debugging tools")]
    public bool DEBUGGING = false;
    public float debug_pubTime = 0;
    public float debug_lisTime = 0;
    public LineRenderer debug_lineRenderer;
    public Material debug_red;
    public Material debug_yellow;
    //public Vector3 lin;
    //public Vector3 ang;
    //public Vector3 direction;
    // Start is called before the first frame update

//  [Header("Static Reward Components")]
    private static float displacementScalar = 0f;//15f;
    private static float veloScalar = 0f;//25f;
    private static float distanceScalar = 5f;//1f;
    private static float spinScalar = 1f;//0.25f/8f;
    private static float ballsScalar = 1f;//0.25f/8f;
    
    private static float scalar = 1f; // scaling up numbers from the PPO because they are toooo small

    [Header("Dynamic Reward Components")]
    private Contact contact = Contact.NoWall;
    private float contactTimeLimit;
    public int ballsTouched;
    public Transform target;
    public Vector3 targetPos;
    public Vector3 previousPosition;
    public Vector3 previousHeading;
    // public BoolMsg terminatedMsg;

    // MESSAGING //
    [Header("For ROS Messages")]
    ROSConnection publishReward;
    ROSConnection publishDone;


    public const float ENFORCED_LATENCY = 0;//*.28f;
    public float enforcedLatencyTimer;
    // public float stepCount;
    public string WHEEL_TOPIC;
    public string REWARD_TOPIC;
    public Float32Msg rewardMsg;

    [Header("Robot Control")]
    private ArticulationBody chassisControl, leftWheel, rightWheel;
    public float speed = 2000f;
    public float rightVelocity;
    public float leftVelocity;
    public Vector3 startPos;
    public Quaternion startDirection;
    public float targetDistInitial;
    public float forwardsBackwards;
    public float leftRight;
    public MeshCollider chassisCollider;

    // public float displacementHighScore;

    
    void Awake()
    {
        leftWheel = transform.Find("left_wheel_link").gameObject.GetComponent<ArticulationBody>();
        rightWheel = transform.Find("right_wheel_link").gameObject.GetComponent<ArticulationBody>();
        chassisControl = gameObject.GetComponent<ArticulationBody>();
        // ROSConnection.GetOrCreateInstance().Subscribe<BoolMsg>("reset", RESTART_SCENE); 
        // TODO: make parent object that tells all the scenes to reset themselves
        //stepCount = 0;
        if (DEBUGGING) debug_lineRenderer = GameObject.Find("PathTracer").GetComponent<LineRenderer>();
        ballsTouched = 0;
    }
    private void Start()
    {
        WHEEL_TOPIC = /*transform.root.name + */"/cmd_vel";
        REWARD_TOPIC = /*transform.root.name +*/ "/reward";

        ROSConnection.GetOrCreateInstance().Subscribe<TwistMsg>(WHEEL_TOPIC, listen);
        //lin = Vector3.zero;
        //ang = Vector3.zero;
        //ROSConnection.GetOrCreateInstance().Subscribe<TwistMsg>("/command/Reinforcement", move);

        // terminatedMsg.data = false;
        // publishTermination = ROSConnection.GetOrCreateInstance();
        // publishTermination.RegisterPublisher<BoolMsg>("/terminated");
        rewardMsg.data = 0f;
        publishReward = ROSConnection.GetOrCreateInstance();

        print(REWARD_TOPIC);
        publishReward.RegisterPublisher<Float32Msg>(REWARD_TOPIC);

        targetPos = target.transform.position;
        startDirection = transform.rotation;
        startPos = transform.position;
        targetDistInitial = Vector3.Distance(startPos, targetPos);

        chassisControl.enabled = true;
        previousPosition = chassisControl.transform.position;
        previousHeading = chassisControl.transform.forward;

        publishDone = ROSConnection.GetOrCreateInstance();
    }

    public void BALL_TOUCHED()
    {
        ballsTouched++;
        targetDistInitial = Vector3.Distance(transform.position, target.transform.position);
    }

    void listen(TwistMsg msg)
    {
        /* The calculations here are the same as the calculations below, in theory. 
         * 
         * rightVelocityLinear = (float)msg.linear.x * dashSpeed;
         * leftVelocityLinear = (float)msg.linear.x * dashSpeed;
         * rightVelocityAngular = -(float)(msg.angular.z) * turnSpeed;
         * leftVelocityAngular = (float)(msg.angular.z) * turnSpeed;
         * rvel = rightVelocityLinear + rightVelocityAngular;
         * lvel = leftVelocityLinear + leftVelocityAngular;
         * rightGoal += (rvel > topSpeed) ? topSpeed : rvel;
         * leftGoal += (lvel > topSpeed) ? topSpeed : lvel;
         * 
         */
        //lin.x = (float)msg.linear.x;
        //ang.z = (float)msg.angular.z;
        //Vector3.Angle(lin, ang);
        forwardsBackwards = (float)msg.linear.x;
        if(forwardsBackwards < 0)
        {
            forwardsBackwards /= 2;
        }
        leftRight = (float)msg.angular.z;

        rightVelocity = (forwardsBackwards * speed + -leftRight * speed)* scalar;
        leftVelocity = (forwardsBackwards * speed + leftRight * speed)* scalar;

        // rightVelocity = (rightVelocity > speed) ? speed : (rightVelocity < -speed) ? -speed : rightVelocity;
        // leftVelocity = (leftVelocity > speed) ? speed : (leftVelocity < -speed) ? -speed : leftVelocity;
        if(Math.Abs(rightVelocity) > speed)
        {
            if (rightVelocity > 0f)
            {
                rightVelocity = speed;
            }
            else if (rightVelocity < 0f)
            {
                rightVelocity = -speed;
            }
        }
        if(Math.Abs(leftVelocity) > speed)
        {
            if (leftVelocity > 0f)
            {
                leftVelocity = speed;
            }
            else if (leftVelocity < 0f)
            {
                leftVelocity = -speed;
            }
        }
        
        previousPosition = chassisControl.transform.position;
        previousHeading = chassisControl.transform.forward;

        enforcedLatencyTimer = 0;
       
        
        if (WHEEL_TOPIC == "env_0/vel_wheels" && DEBUGGING)
        {
            debug_lisTime = Time.time;
            debug_lineRenderer.material = debug_yellow; 
            Debug.Log("Sub Latency Time = \t\t\tS| " + (debug_lisTime - debug_pubTime));
            //Debug.Log("debug_deltaLP = \t\t\tLP| " + debug_deltaListenPublish);
        }
        
    }
    void publish()
    {

        if(contact == Contact.Wall)
        {
            // switch this back later
            rewardMsg.data -= (contactTimeLimit);   
        }
        if (contact == Contact.NoWall)
        {


            /* Heading Math
            * Used to scale the Displacement reward, in the displacementVSheading variable
            *                  | Heading Towards | Heading Away | (from target)
            * Moving Towards   |       +         |       -      | aka displacementDelta+
            * Moving Away      |       0         |       -      | aka displacementDelta-
            */
            targetPos = target.transform.position;
          
            float displacementDelta = Vector3.Distance(previousPosition, targetPos) - Vector3.Distance(chassisControl.transform.position, targetPos);
            Vector3 lineToTarget = targetPos - chassisControl.transform.position;
            float headingRelativeToTarget = Vector3.Angle(lineToTarget, chassisControl.transform.forward);
            float headingNormalized = 2f * (0.5f - (headingRelativeToTarget / 180f));

            // If moving backwards but also facing the target, you don't get any reward or punishment
            int movingAwayWhileHeadingForwards = (displacementDelta < 0f && headingNormalized > 0.0f) ? 0 : 1;
            // print("Normalized heading is " + headingNormalized);
            float displacementVSheading = (headingNormalized > 0 && displacementDelta > 0) ? 1 : -1;
            float displacementReward = displacementScalar * displacementVSheading * movingAwayWhileHeadingForwards * Math.Abs(displacementDelta * headingNormalized);

            // Spinmath - reward the robot for spinning in the right direction
            // If current heading is more towards the target than previous heading, reward accordingly
            // headingNormalized is an angle from 1 (pointing right towards it) and -1 (pointing right away)
            // Therefore, we should want heading to be Bigger
            //  float previousHeadingRelativeToTarget = Vector3.Angle(previousHeading, lineToTarget);
            //  float previousHeadingNormalized = 2f * (0.5f - (previousHeadingRelativeToTarget / 180f));

            // pos or neg depending on if you're going the right way
            // int turningTheRightWay = (headingNormalized > previousHeadingNormalized) ? 1 : -1;
            // print("previousHeadingNormalized is " + previousHeadingNormalized);

            //  float spinReward = spinScalar * (headingNormalized + previousHeadingNormalized); // -2>+2, -2 when turned fully away 2 steps, and 2 when going straight ahead 2 steps
            float spinReward = spinScalar * headingNormalized;
            // if previously heading right way and continuing to go right way, get 2
            // if going wrong way and continuing to go wrong way, get -2
            // if going left then going forwards get 1


            // Velocity Math...
            float velo = chassisControl.velocity.magnitude;
            float veloReward = veloScalar *velo;

            // Distance Math
            float distanceFromTarget = Vector3.Distance(chassisControl.transform.position, target.position);
            float distanceTravelledToTarget = targetDistInitial - distanceFromTarget;
            float distanceTravelledToTargetNormalized = distanceTravelledToTarget / targetDistInitial;
            float distanceReward = distanceTravelledToTargetNormalized * distanceScalar;

            float rewardForBallTouching = ballsTouched * ballsScalar;

            // spin scalar is 0.25f/8f
            // displacement scalar is 8f
            rewardMsg.data = ( displacementReward + veloReward + distanceReward + spinReward + rewardForBallTouching);
            
            /*
            if (WHEEL_TOPIC == "env_0/vel_wheels")
            {
                Debug.Log("Displacement from last step: " + displacementDelta);

                Debug.Log("\t Reward\t\t\t\t: " + rewardMsg.data);
                Debug.Log("\t Reward from displacement\t: " + displacementScalar * displacementDelta);

                Debug.Log("\tVelocity\t\t\t\t: " + velocity);
                Debug.Log("\t Reward from velo\t\t: " + veloScalar * velocity);

                Debug.Log("\tTotal displacement\t\t: " + displacementTotal);
                Debug.Log("\tReward from distance\t\t: " + totalScalar * Math.Max(displacementTotal - displacementHighScore, 0));
            }
             */
        }
        // swtich this back later
        if (rewardMsg.data < -10f)
        {
            rewardMsg.data = -10f;
        }
        // publishTermination.Publish("/terminated", terminatedMsg);
        
        if (WHEEL_TOPIC == "env_0/vel_wheels" && DEBUGGING)
        {
            debug_pubTime = Time.time;
            debug_lineRenderer.material = debug_red;
            Debug.Log("Pub Latency Time = \t\t\tP| " + (debug_pubTime-debug_lisTime));
            //Debug.Log("debug_deltaLP = \t\t\tLP| " + debug_deltaListenPublish);
        }
        
        publishReward.Publish(REWARD_TOPIC, rewardMsg);
    }

    public void RESTART_SCENE()
    {
        forwardsBackwards = 0;
        leftRight = 0;
        ballsTouched = 0;
        contactTimeLimit = 0;
        rightVelocity = 0;
        leftVelocity = 0;
        rewardMsg.data = 0; 
        targetDistInitial = Vector3.Distance(transform.position, target.transform.position);
        chassisControl.TeleportRoot(startPos, startDirection);
        leftWheel.SetDriveTargetVelocity(ArticulationDriveAxis.X, 0);
        rightWheel.SetDriveTargetVelocity(ArticulationDriveAxis.X, 0);
        contact = Contact.NoWall;
        contactTimeLimit = 0;
        targetPos = target.position;
        //stepCount = 0;

        //rightGoal = 0;
        //leftGoal = 0;
        // terminatedMsg.data = true;
        //leftWheel.SetDriveTarget(ArticulationDriveAxis.X, 0);
        //rightWheel.SetDriveTarget(ArticulationDriveAxis.X, 0);
        // chassisControl.TeleportRoot(startPos, startDirection);
    }

    // ForgivenessTime will tick between the robot touching the wall and the robot being counted as no longer touching the wall.
    private static float forgivenessTimeMax = 1f;
    private float forgivenessTime = forgivenessTimeMax;
    private void OnTriggerStay(Collider other)
    {
        // Logic: Because contactTimeLimit is updated every frame, Adding some value to it will tell us whether or not it changed.
        // HOWEVER: Sometimes the robot thinks it gets free from the walls for an infinitessimally small time.
        //      In this case, the robot will think it's gotten away from the wall when it hasn't frfr.
        //      Therefore, the change in this value to check if we're actually changing over time
        float previousTimeLimit = contactTimeLimit;     // Track previous time limit
        if (other.tag == "Respawn")                     // If you're touching a wall
        {
            forgivenessTime = forgivenessTimeMax;                       // then no forgiveness
            contact = Contact.Wall;                     // Wall!
            contactTimeLimit += Time.deltaTime;         // Time passes
        }
        if (previousTimeLimit == contactTimeLimit)      // If contact time is the same as before, then we haven't touched anything
        {
            // This is more of a sanity check than a functional thing. 
            if (contactTimeLimit > 0f)                  // contactTimeLimit should only go up, until the end of this statement, so this is logically equivalent to the outer if condition
            {
                forgivenessTime -= Time.deltaTime;     
            }
            if (forgivenessTime <= 0)                   
            {
                contact = Contact.NoWall;
                contactTimeLimit = 0f;
                forgivenessTime = forgivenessTimeMax;
            }
        }
        /*
        if (contactTimeLimit > 3)
        {
            restartScene();
            terminatedMsg.data = false;
        }
        */
    }
    // public int publishTime = 0;
    private void FixedUpdate()
    {

        if (enforcedLatencyTimer>=ENFORCED_LATENCY)
        {
            publish();
        }
        else
        {
            enforcedLatencyTimer += Time.deltaTime;
        }

        /*
        if (publishTime >= 2)
        {
            publishTime = 0;
        }
        else publishTime += 1;
         */
        /*
        rightGoal += rightVelocity;
        leftGoal += leftVelocity;
        leftWheel.SetDriveTarget(ArticulationDriveAxis.X, leftGoal);
        rightWheel.SetDriveTarget(ArticulationDriveAxis.X, rightGoal);
         */
        
        if (chassisControl.angularVelocity.magnitude > 0.5f)
        {
            if (Math.Abs(rightVelocity) > speed / 2)
            {
                if (rightVelocity > 0f)
                {
                    rightVelocity = speed / 2;
                }
                else if (rightVelocity < 0f)
                {
                    rightVelocity = -speed / 2;
                }
            }
            if (Math.Abs(leftVelocity) > speed / 2)
            {
                if (leftVelocity > 0f)
                {
                    leftVelocity = speed / 2;
                }
                else if (leftVelocity < 0f)
                {
                    leftVelocity = -speed / 2;
                }
            }
        }
        if(chassisControl.velocity.magnitude > 0.3f)
        {
            if(rightVelocity < -speed / 2)
            {
                rightVelocity = -speed / 2;
            }
            if(leftVelocity < -speed / 2)
            {
                leftVelocity = -speed / 2;
            }
        }
        leftWheel.SetDriveTargetVelocity(ArticulationDriveAxis.X, leftVelocity);
        rightWheel.SetDriveTargetVelocity(ArticulationDriveAxis.X, rightVelocity);
    }
}

    /*
    void restartScene()
    {
        rightVelocity = 0;
        leftVelocity = 0;
        //rightGoal = 0;
        //leftGoal = 0;
        terminatedMsg.data = true;
        rewardMsg.data -= 100;
        chassisControl.TeleportRoot(startPos, startDirection);
        //leftWheel.SetDriveTarget(ArticulationDriveAxis.X, 0);
        //rightWheel.SetDriveTarget(ArticulationDriveAxis.X, 0);
        leftWheel.ResetInertiaTensor();
        rightWheel.ResetInertiaTensor();
        chassisControl.ResetInertiaTensor();
        // chassisControl.TeleportRoot(startPos, startDirection);
        contactTimeLimit = 0;
    }
     */