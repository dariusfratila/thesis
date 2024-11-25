import React from "react";
import styles from "./PredictionsDisplay.module.css";

const PredictionsDisplay = React.forwardRef(({ words, probabilities }, ref) => {
    return (
        <div ref={ref} className={styles.lipReadOutputContainer}>
            <h2>Predicted Words and Probabilities:</h2>
            <div className={styles.resultsContainer}>
                {words.map((word, index) => (
                    <div key={index} className={styles.resultItem}>
                        <span className={styles.word}>{word.toUpperCase()}</span>
                        <span className={styles.probability}>
                            {`${(probabilities[index] * 100).toFixed(3)}%`}
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
});

export default PredictionsDisplay;
