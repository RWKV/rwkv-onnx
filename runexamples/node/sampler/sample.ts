/**
 * Given the float array, get its max value.
 * @param {Float32Array} arr 
 * @returns {number}
 */
export function getMaxFloat(arr:Float32Array): number {
	let max = -Infinity;
	for (let i = 0; i < arr.length; i++) {
		if (arr[i] > max) max = arr[i];
	}
	// if( max == 0.0 ) {
	// 	throw "Unexpected max == 0 in getMaxFloat"
	// }
	return max;
}

/**
 * Implements the softmax function.
 * Which reads the input array, and outputs an array of
 * index to probability mappings.
 * 
 * @param {Float32Array} arr to read from
 * @returns 
 */
function softmaxToProbPair( arr:Float32Array ) {
	// Get the logits size
	const logits_size = arr.length;

	// Setup the probability pair array
	const probPair = new Array(logits_size);

	// Get the max value
	const max = getMaxFloat(arr);

	// Subtract the max value from each element
	// and calculate the sum of exponents
	let sum = 0.0;
	for (let i = 0; i < logits_size; i++) {
		const prob = Math.exp(arr[i] - max);
		probPair[i] = [i, prob];
		sum += prob;
	}

	// Divide each element by the sum
	for (let i = 0; i < logits_size; i++) {
		probPair[i][1] = probPair[i][1] / sum;
	}

	// Return the sorted probability pair
	return probPair.sort((a, b) => b[1] - a[1]);
}

/**
 * sample_logits operation, used to decide on the next token
 * given the current logits output.
 * 
 * @param {Float32Array} logits - The logits output from the model
 * @param {number} temp - The temperature to use for sampling
 * @param {number} top_p - The top_p to use for sampling
 * 
 * @returns {Object} containing the token index, and the final logits
 */
export function sampleLogits(logits:Float32Array, temp: number = 1.0, top_p: number = 1.0): {token: number, logprobs: number[][]} {
	//
	// !!! Important note !!!
	//
	// Because sampling function can differ between implementation.
	// We are following blinks RWKV_in_150_lines as close as possible here.
	// To help ensure consistency between implementations of RWKV
	//
	// https://github.com/BlinkDL/ChatRWKV/blob/main/RWKV_in_150_lines.py#L119
	//
	// This will differ from minGPT implementation (and some other implementations)
	// https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L283
	//

	// Validate the logits buffer
	if (logits == null) {
		throw "Invalid logits buffer";
	}

	// If temp is 0.0, then we just return the max logit index
	if (temp <= 0.0) {
		return {token:logits.indexOf(getMaxFloat(logits)), logprobs:[]};
	}

	// Validate the top_p
	if (top_p < 0.0) {
		throw "Invalid top_p";
	}

	// Normalize temp, and top_p as float values
	temp = temp * 1.0;
	top_p = top_p * 1.0;

	// Change into a list of [index, prob] pairs
	// while applying softmax at the same time
	let probPairs = softmaxToProbPair(logits);

	// Get the cumulative probability pre and post temp scaling
	let cumSoftmaxProb = 0.0;
	let cumTempProb = 0.0;
	for (let i = 0; i < probPairs.length; i++) {
		const tempProb = Math.pow(probPairs[i][1], 1.0 / temp);
		cumSoftmaxProb += probPairs[i][1];
		cumTempProb += tempProb;
		probPairs[i][1] = tempProb;

		// Top_p filtering
		// ---
		// If top_p is is valid and
		// If we have reached the top_p threshold, then break
		// This is done here to avoid the need to loop again
		if (top_p < 1.0 && cumTempProb >= top_p) {
			probPairs = probPairs.slice(0, i + 1);
			break;
		}
	}
	
	// Time to sample 
	let randProb = Math.random() * cumTempProb;

	// Find the index of the sampled token
	for(let i = 0; i < probPairs.length; i++) {
		randProb -= probPairs[i][1];
		if (randProb <= 0.0) {
			return {
				token: probPairs[i][0],
				logprobs: probPairs
			};
		}
	}

	// Out of bound? return the first index
	// (higest probability token)
	//
	// This should not happen unless an extream case 
	// of floating point accuracy error
	return {
		token: probPairs[0][0],
		logprobs: probPairs
	};
}

