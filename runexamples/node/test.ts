// Declare to typescript compiler that we are using onnxruntime-node and to ignore type checking.

// Ignore type checking for onnxruntime-node.
// @ts-ignore
import * as ort from 'onnxruntime-node';

import {InferenceSession, Tensor, TypedTensor, InferenceSessionFactory} from 'onnxruntime-common';


type WkvStateName = "wkv"|""
type StateKey = `instate${WkvStateName}${number}`

type State = {
    
    [key: StateKey]: TypedTensor<"float32">
}

type TokenJob = {
    token: number
    state: State
    callback: (logits:Tensor,state:State) => void
}

// function zipState(states:State[]):State {
//     const result:State = {}
//     for (const key in states[0]) {
//         const tensors = states.map(state => state[key])
//         // result[key] = Tensor.concat(tensors)
//     }
//     return result
// }


class RWKV<Initialized extends boolean = false> {

    embed:number = 0
    layers:number = 0
    heads:number = 0
    model: Initialized extends true ? InferenceSession : null = null as Initialized extends true ? InferenceSession : null
    path : string 
    jobQueue:TokenJob[] = []
    currentJobs:TokenJob[] = []
    stateHolder:State = {}

    constructor(path:string) {
        this.path = path
    }
    
    
    unzipState(state:State):State[] {
        const result:State[] = []

        const B = state.instate0.dims[0]

        for (let i = 0; i < B; i++) {
            const newState:State = {}
            for (const key in state) {
                const tensor:TypedTensor<"float32"> = state[key as StateKey]
                const dims = tensor.dims
                const muldims = dims.slice(1).reduce((a,b) => a*b)
                const data = tensor.data.slice(i*muldims,(i+1)*muldims)
                const ten = new Tensor(data,[1,...dims.slice(1)])
                newState[key as StateKey] = ten
            }
            result.push(newState)
        }
        return result
    }

    zipState(states:State[]) {
        for (const key in states[0]) {
            const tensors = states.map(state => (state[key as StateKey] as TypedTensor<"float32">))
            const dims = tensors[0].dims
            const newdims = [states.length,...dims.slice(1)];
            const newsize = newdims.reduce((a,b) => a*b)

            if(this.stateHolder[key as StateKey] == undefined ){
                this.stateHolder[key as StateKey] = new Tensor("float32",new Float32Array(newsize),newdims)
            }

            if (this.stateHolder[key as StateKey].dims[0] != states.length) {
                this.stateHolder[key as StateKey] = new Tensor("float32",new Float32Array(newsize),newdims)
            }
            
            
            for (let i = 0; i < states.length; i++) {
                const state = states[i];
                const tensor = state[key as StateKey]
                const data = tensor.data
                this.stateHolder[key as StateKey].data.set(data,i*data.length)
            }

        // 390 18
        }
    }

    async run (){
        if (this.jobQueue.length > 0) {
            const jobs = this.jobQueue.splice(0,Math.min(this.jobQueue.length,128))
            
            this.currentJobs = jobs
            const states = jobs.map(job => job.state)
            this.zipState(states)
            const tokens = new Tensor("int32",jobs.map(job => job.token),[jobs.length])
            const inputnamesreversed:StateKey[] = this.model!.inputNames as StateKey[]
            const inputnames = inputnamesreversed.reverse()
            
            const outputnamesReversed = this.model!.outputNames as string[];
            const outputnames = outputnamesReversed.reverse()
            // console.log("outputnames: ", outputnames)

            const currenttime = Date.now()

            const outputs = await this.model!.run({"input0":tokens,...this.stateHolder});
            
            console.log("Concurrent Jobs: ", jobs.length, " Time: ", Date.now()-currenttime)

            const splits = outputnames.reduce((a,b) => ({
                "logits": outputs[b].dims[1] == Math.pow(2,16) ? [...a.logits,outputs[b] as TypedTensor<"float32">] : a.logits,
                "instate": outputs[b].dims.length == 2 && outputs[b].dims[1] != Math.pow(2,16) ?[...a.instate,outputs[b] as TypedTensor<"float32">] : a.instate,
                "instatewkv": outputs[b].dims.length == 4 ?[...a.instatewkv,outputs[b] as TypedTensor<"float32">] : a.instatewkv
            }), {
                "logits": [] as TypedTensor<"float32">[],
                "instate": [] as TypedTensor<"float32">[],
                "instatewkv": [] as TypedTensor<"float32">[]
            })
            

            const logits = splits.logits[0] as Tensor

            // console.log("logits: ", splits.logits.length)
            // console.log("instate: ", splits.instate.length)
            // console.log("instatewkv: ", splits.instatewkv.length)

            const nextInputState = {} as State

            for (let i = 0; i < splits.instate.length; i++) {
                const key = "instate"+i as StateKey;
                nextInputState[key] = splits.instate[i]
            }

            for (let i = 0; i < splits.instatewkv.length; i++) {
                const key = ("instatewkv"+i) as StateKey;
                nextInputState[key] = splits.instatewkv[i]
            }

            const newstates = this.unzipState(nextInputState)

            this.stateHolder = {}

            newstates.forEach((state,i) => {
                // console.log("state: ", Object.keys(state))
                jobs[i].callback(logits,state)
            })
            this.currentJobs = []
            
        }

    }

      


    async load():Promise<RWKV<true>> {
        const sess:InferenceSession = await (ort.InferenceSession as InferenceSessionFactory).create('../../RWKV_32_2560_32_15_QUInt8-pc-norr-ext.onnx', {
            interOpNumThreads: 8,
            intraOpNumThreads: 8,
            executionMode: 'parallel',
            executionProviders: ["cpu"]
        });

        // prepare inputs. a tensor need its corresponding TypedArray as data
        const inputnames = sess.inputNames;

        // console.log("inputnames: ",inputnames)

        // Get the shape of the input
        
        this.embed = 2560
        this.layers = (inputnames.length-1)/3
        this.heads = 40 // 32 if 1b5

        // console.log("embed: ", this.embed)
        // console.log("layers: ", this.layers)

        this.model = sess as Initialized extends true ? InferenceSession : null

        return this as unknown as RWKV<true>
    }

    newState():State {
        const result:State = {}
        for (let i = 0; i < this.layers*2; i++) {
            result[`instate${i}` as StateKey] = new Tensor("float32",new Float32Array(this.embed),[1,this.embed])
        }

        for (let i = 0; i < this.layers; i++) {
            result[`instatewkv${i}` as StateKey] = new Tensor("float32",new Float32Array(((this.embed * this.embed)/this.heads)),[1,this.heads,this.embed/this.heads, this.embed/this.heads])
        }
        return result
    }


}



// use an async context to call onnxruntime functions.
async function main() {
    try {
        // create a new session and load the specific model.
        //
        // the model in this example contains a single MatMul node
        // it has 2 inputs: 'a'(float32, 3x4) and 'b'(float32, 4x3)
        // it has 1 output: 'c'(float32, 3x3)

        // const testTensor:Tensor = new Tensor('float32', [0, 0, 0, 0], [1,2,2]);

        // console.log("testTensor: ", testTensor)
        
        // console.log("testTensor.data: ", testTensor.data.slice(0,2))

        const model = await new RWKV('../../RWKV_32_2560_32_15_QUInt8-pc-norr-ext.onnx').load()
        
        // run model.run every 100ms
        setInterval(() => {
            model.run()
        }, 100);

        const pushToken = (stuff:number, state:State) => {
            // console.log("stuff: ", stuff)

            model.jobQueue.push({
                token: 0,
                state,
                callback: (logits,state) => {
                    pushToken(stuff+1,state)
                }
            })
        }

        // every 3 seconds, push another token

        setInterval(() => {
            pushToken(0,model.newState())
        }
        , 3000);

        // const state = model.newState()


        while (true) {
            await new Promise(resolve => setTimeout(resolve, 1000));
        }

        


    } catch (e) {
        console.error(`failed to inference ONNX model: ${e}.`);
    }
}

main();