package model;

import java.util.Random;

public abstract class Worker {
	
	private String name;
	private int age;
	protected Random random = new Random();
	
	public abstract void work();
	
	public void getUp() {
		System.out.println(name + " bangun");
	}
	
	public void goWork() {
		System.out.println(name + " is going to work");
	}
	
	public void goSleep() {
		System.out.println(name + " is sleep");
	}
	
	public abstract void showEarn();
	
	public void dailyRoutine() {
		getUp();
		goWork();
		work();
		goSleep();
		showEarn();
	}
	
	
	public Worker(String name, int age) {
		super();
		this.name = name;
		this.age = age;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public int getAge() {
		return age;
	}

	public void setAge(int age) {
		this.age = age;
	}
	
	
}
