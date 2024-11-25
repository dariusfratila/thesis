import React, { useState } from "react";
import styles from "./Faq.module.css";

function Faq() {
  const [selectedQuestion, setSelectedQuestion] = useState(null);

  const toggleQuestion = (index) => {
    setSelectedQuestion((prevSelectedQuestion) => {
      return prevSelectedQuestion === index ? null : index;
    });
  };

  const questionsAnswers = [
    {
      question: "What's the main focus of the lip-reading research?",
      answer:
        "The research focuses on improving lip reading accuracy using techniques like TCN and cross-lingual domain adaptation.",
    },
    {
      question: "How does the model improve lip reading?",
      answer:
        "The model uses a combination of techniques including TCN for capturing temporal dependencies and cross-lingual domain adaptation for leveraging data from other languages, which together improve lip reading accuracy.",
    },
    {
      question: "What is Temporal Convolutional Network (TCN)?",
      answer:
        "TCN is a type of neural network designed for sequence modeling. It uses dilated convolutions and gating mechanisms to capture long-term dependencies in time series data.",
    },
    {
      question:
        "Can the techniques from this research be applied to other languages?",
      answer:
        "Yes, the techniques used in this research, especially cross-lingual domain adaptation, can be applied to lip reading in other languages, potentially improving performance on underrepresented datasets.",
    },
    {
      question: "Is the technology able to recognize speech in real-time?",
      answer:
        "Real-time recognition is a complex challenge, but the research aims to optimize algorithms for efficient processing, which is a step towards real-time application.",
    },
  ];

  return (
    <div className={styles.faqContainer}>
      <h1 className={styles.faqTitle}>Frequently asked questions</h1>
      <div className={styles.faq}>
        {questionsAnswers.map((item, index) => (
          <div
            key={index}
            className={`${styles.faqItem} ${
              selectedQuestion === index ? styles.active : ""
            }`}
          >
            <button
              className={styles.questionButton}
              onClick={() => toggleQuestion(index)}
            >
              {item.question}
              <span
                className={`${styles.chevron} ${
                  selectedQuestion === index ? styles.rotate : ""
                }`}
              >
                &#9662;
              </span>{" "}
              {}
            </button>
            <div
              id={`answer-${index}`}
              className={`${styles.answer} ${
                selectedQuestion === index ? styles.show : ""
              }`}
            >
              {item.answer}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default Faq;
