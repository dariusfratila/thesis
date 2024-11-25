import React from "react";
import styles from "./DetailedApproach.module.css";

function DetailedApproach() {
  const details = [
    {
      title: "Data Augmentation Techniques",
      description:
        "Data augmentation techniques are essential for improving the performance of lip-reading models by artificially increasing the diversity of the training dataset. Methods such as random cropping, rotation, flipping, and adding noise help the model become more robust to variations in input data, leading to better generalization and accuracy.",
    },
    {
      title: "Lateral Inhibition for Focus Enhancement",
      description:
        "Lateral inhibition, mimicking biological neural processes, allows the network to suppress less important features while amplifying the most relevant ones. This technique is crucial for improving the clarity of visual features in lip reading, especially in low-light conditions.",
    },
    {
      title: "Temporal Convolutional Network",
      description:
        "Temporal Convolutional Networks (TCNs) are designed for sequence modeling. They use dilated convolutions and gating mechanisms to capture long-term dependencies in time series data. The multi-scale aspect of TCN refers to the use of convolutional layers with different kernel sizes, capturing features at different time scales.",
    },
  ];

  return (
    <section className={styles.container}>
      <h2 className={styles.heading}>
        A Closer Look at the Lip-Reading Research
      </h2>
      {details.map((detail, index) => (
        <article key={index} className={styles.detailSection}>
          <h3 className={styles.detailTitle}>{detail.title}</h3>
          <p className={styles.detailDescription}>{detail.description}</p>
        </article>
      ))}
    </section>
  );
}

export default DetailedApproach;
