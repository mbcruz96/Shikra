import jsonlines
import json
import os

if __name__=="__main__":
	prediction_file = "multitest_COCO_POPE_RANDOM_q_a_extra_prediction.jsonl"
	true_pos = 0
	false_pos = 0
	true_neg = 0
	false_neg = 0
	tot_pos = 0
	num_samples = 0

	with jsonlines.open(prediction_file, 'r') as infile:
		for entry in infile:
			answer = entry["pred"].strip().lower()
			prediction = entry["target"]
			index = prediction.find("ASSISTANT:")
			prediction = prediction[index:].strip().lower()

			if "yes" in answer:
				ans = 1
			else:
				ans = 0
			if "yes" in prediction:
				pred = 1
			else:
				pred = 0

			if pred == 1 and ans == 1:
				true_pos += 1
				tot_pos += 1
			elif pred == 1 and ans == 0:
				false_pos += 1
				tot_pos += 1
			elif pred == 0 and ans == 1:
				false_neg += 1
			else:
				true_neg += 1
			num_samples += 1

	precision = true_pos / (true_pos + false_pos)
	recall = true_pos / (true_pos + false_neg)
	f1_score = 2 * precision * recall / (precision + recall)
	yes = tot_pos / num_samples

	with open("multitest_COCO_POPE_RANDOM_metrics.txt", 'w') as outfile:
		outfile.write(f"Precision: {precision:.4f}\n")
		outfile.write(f"Recall: {recall:.4f}\n")
		outfile.write(f"F1-score: {f1_score:.4f}\n")
		outfile.write(f"Yes: {yes:.4f}\n")
