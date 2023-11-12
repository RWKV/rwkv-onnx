// Declare to typescript compiler that we are using onnxruntime-node and to ignore type checking.

// Ignore type checking for onnxruntime-node.
// @ts-ignore
import * as ort from 'onnxruntime-node';

import {InferenceSession, Tensor, TypedTensor, InferenceSessionFactory} from 'onnxruntime-common';
import { WorldTokenizer } from './tokenizer/tokenizer';
import { sampleLogits } from './sampler/sample';

// write data content to a file continously
import { createWriteStream } from 'fs';

type WkvStateName = "wkv"|""
type StateKey = `state${WkvStateName}${number}`
type OutStateKey = `state${WkvStateName}${number}out`

type State = {
    
    [key: StateKey]: TypedTensor<"float32">
}

type TokenJob = {
    token: number
    state: State
    callback: (logits:Tensor,state:State) => void
}

type ProbState = {
    probs: Tensor,
    state: State
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
    
    
    unzipState(state:State, oldStates:State[]):State[] {
        const result:State[] = []

        const B = oldStates.length

        for (let i = 0; i < B; i++) {
            for (const key in state) {
                const tensor:TypedTensor<"float32"> = state[key as StateKey]
                const dims = tensor.dims
                const muldims = dims.slice(1).reduce((a,b) => a*b)
                const data = tensor.data.subarray(i*muldims,(i+1)*muldims)
                oldStates[i][key as StateKey].data.set(data)
            }
        }
        return oldStates
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
           
            const stateNames = this.model!.inputNames.filter(name => name.startsWith("state")) as StateKey[]
            
            const outputs = await this.model!.run({"input0":tokens,...this.stateHolder});
    
            

            const logits = Object.values(outputs).find(
                (tensor) => tensor.dims[1] == 2**16
            ) as Tensor

            const nextInputState = stateNames.reduce((acc,name) => {
                acc[name] = outputs[name+"out"] as TypedTensor<"float32">
                return acc
            }, {} as State)




            const newstates = this.unzipState(nextInputState, states)

            this.stateHolder = {}

            newstates.forEach((state,i) => {
                // console.log("state: ", Object.keys(state))
                jobs[i].callback(logits,state)
            })
            this.currentJobs = []
            
        }

    }

      


    async load():Promise<RWKV<true>> {
        const sess:InferenceSession = await (ort.InferenceSession as InferenceSessionFactory).create(this.path, {
            interOpNumThreads: 8,
            intraOpNumThreads: 8,
            executionMode: 'parallel',
            executionProviders: ["cpu"]
        });

        // prepare inputs. a tensor need its corresponding TypedArray as data
        const inputnames = sess.inputNames;

        // console.log("inputnames: ",inputnames)

        // Get the shape of the input
        
        this.embed = 2048
        this.layers = (inputnames.length-1)/3
        this.heads = 32 // 32 if 1b5

        // console.log("embed: ", this.embed)
        // console.log("layers: ", this.layers)

        this.model = sess as Initialized extends true ? InferenceSession : null

        return this as unknown as RWKV<true>
    }

    newState():State {
        const result:State = {}
        for (let i = 0; i < this.layers*2; i++) {
            result[`state${i}` as StateKey] = new Tensor("float32",new Float32Array(this.embed),[1,this.embed])
        }

        for (let i = 0; i < this.layers; i++) {
            result[`statewkv${i}` as StateKey] = new Tensor("float32",new Float32Array(((this.embed * this.embed)/this.heads)),[1,this.heads,this.embed/this.heads, this.embed/this.heads])
        }
        return result
    }

    async readContext(context:number[],state?:State):Promise<ProbState>{
        return new Promise((resolve,reject) => {
        if (state == undefined) {
            state = this.newState()
        }
        
        var current = 0;

        const callback = (logits:Tensor,nextstate:State) => {
            current+=1
            console.log("current: ",current, nextstate.state1.data[0])
            if (current == context.length) {
                resolve({
                    state: nextstate,
                    probs: logits
                })
            }else{
                this.jobQueue.push({
                    token: context[current],
                    state: nextstate,
                    callback
                })
            }
        }

        this.jobQueue.push({
            token: context[current],
            state,
            callback
        })

        
    })

}

    startListening() {
        setInterval(() => {
            this.run()
        }, 10);
    }

    sampleLogits(logits:Tensor):number {
        return sampleLogits(logits.data as Float32Array,0.9,0.9).token
    }

    async genToken(input:ProbState, callback:(i:number)=>void):Promise<ProbState>{
        const {state,probs} = input
        const token = this.sampleLogits(probs)
        callback(token)
        const newstate = await this.readContext([token],state)
        return newstate
    }

    

}



// use an async context to call onnxruntime functions.
async function main() {
    try {

        const tokens = WorldTokenizer.encode("\n### Instruction:\nCan you please write a short epic fantasy story? ### Response: \n")

        // create a new session and load the specific model.
        //
        // the model in this example contains a single MatMul node
        // it has 2 inputs: 'a'(float32, 3x4) and 'b'(float32, 4x3)
        // it has 1 output: 'c'(float32, 3x3)

        // const testTensor:Tensor = new Tensor('float32', [0, 0, 0, 0], [1,2,2]);

        // console.log("testTensor: ", testTensor)
        
        // console.log("testTensor.data: ", testTensor.data.slice(0,2))

        const model = await new RWKV('../../RWKV_24_2048_32_15.onnx').load()
        model.startListening()

        const writeStream = createWriteStream("./out.txt")
        
        var curstate = await model.readContext(tokens)

        console.log("curstate: ", curstate.state.state1.data[0])

        for(var i = 0; i < 100; i++){
            curstate = await model.genToken(curstate,
                (token) => {
                    try{

                        const text = WorldTokenizer.decode([token])
                        writeStream.write(text)
                        
                    }
                    catch(e){
                        console.log("error: ", e)
                    }
                }
            )
        }
        
        
        


        while (true) {
            await new Promise(resolve => setTimeout(resolve, 1000));
        }

        


    } catch (e) {
        console.error(`failed to inference ONNX model: ${e}.`);
    }
}

main();